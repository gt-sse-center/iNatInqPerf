"""Functions to help with managing (docker) containers."""

import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager

import docker
from loguru import logger

from inatinqperf.configuration import Config


@contextmanager
def container_context(config: Config | dict) -> Generator[object]:
    """Context manager for running the vector database container.

    If the containers key is not provided in the config, then it executes an empty context,
    allowing for easy optional use of containers.

    Args:
        config (Config): The configuration with the details about the containers.
    """
    containers: list[object] = []
    network = None

    if isinstance(config, dict):
        config = Config(**config)

    containers_config = config.containers

    if not containers_config:
        logger.info("No container configuration provided, not running container(s)")
        yield containers

        # No cleanup, so return
        return

    client = docker.from_env()

    try:
        # We need containers stood up, so first set up the network if specified
        if network_name := config.container_network:
            # If the network already exists, then remove it
            for existing_network in client.networks.list():
                if network_name == existing_network.name:
                    existing_network.remove()

            network = client.networks.create(network_name, driver="bridge")

        for container_cfg in containers_config:
            if container_cfg is None:
                continue

            # If container is already running, then stop and remove it
            # so we can start in a clean environment.
            try:
                existing_container = client.containers.get(container_cfg.name)
                existing_container.stop()
                existing_container.remove(v=True)

            except docker.errors.NotFound:
                pass

            container = client.containers.run(
                image=container_cfg.image,
                name=container_cfg.name,
                hostname=container_cfg.hostname,
                ports=container_cfg.ports,
                environment=container_cfg.environment,
                volumes=container_cfg.volumes,
                command=container_cfg.command,
                security_opt=container_cfg.security_opt,
                healthcheck=container_cfg.healthcheck,
                network=container_cfg.network,
                remove=True,
                detach=True,  # enabled so we don't block on this
            )
            containers.append(container)

            logger.info(f"Running container with image: {container_cfg.image}")

            # Allow clustered services to stabilize before launching the next container.
            time.sleep(5)

        yield containers

    finally:
        logger.info("Cleaning up containers")

        try:
            # Stop containers in reverse order
            for container in containers[::-1]:
                container.stop()

        except Exception as exc:
            logger.warning(f"Failed to stop container: {exc}")

        # Remove network if it was created.
        if network:
            network.remove()

        client.close()


def _is_safe_container_name(name: str) -> bool:
    """Check if the name of the container is safe to run in a cluster."""
    _allowed_container_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return name and all(char in _allowed_container_chars for char in name)


def log_single_container_tail(docker_cmd: str, container_name: str) -> None:
    """Log the output tail of running a single container `container_name` via the `docker_cmd`."""
    if not _is_safe_container_name(container_name):
        logger.warning(f"Skipping unsafe container name: {container_name!r}")
        return

    try:
        result = subprocess.run(  # noqa: S603
            [docker_cmd, "logs", "--tail", "20", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"Failed to fetch logs for container '{container_name}': {exc}")
        return

    output = result.stdout.strip()
    if output:
        logger.warning(f"[{container_name}] {output}")
    error_output = result.stderr.strip()
    if error_output:
        logger.warning(f"[{container_name}][stderr] {error_output}")
