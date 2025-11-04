"""Functions to help with managing (docker) containers."""

from collections.abc import Generator
from contextlib import contextmanager

import docker
from loguru import logger

from inatinqperf.benchmark.configuration import Config


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

        yield containers

    finally:
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
