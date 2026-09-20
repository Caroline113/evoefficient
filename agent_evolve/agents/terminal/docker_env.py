"""Docker environment management for Terminal-Bench 2.x containers."""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class TB2Container:
    """Manages a Terminal-Bench 2.x Docker container lifecycle."""

    def __init__(self, image_name: str, container_name: str | None = None,
                 *, cpus: float | int | None = None,
                 memory_mb: int | None = None,
                 environment: dict | None = None):
        self.image_name = image_name
        self.container_name = container_name or f"tb2-{uuid.uuid4().hex[:12]}"
        self.cpus = cpus
        self.memory_mb = memory_mb
        self.environment = environment or {}
        self._running = False

    def start(self, max_retries: int = 3) -> str:
        """Start the container, verifying it is actually running before returning.

        Retries up to max_retries times on transient startup failures (race
        conditions under parallel load can cause containers to exit immediately
        even when docker run -d reports success).
        """
        for attempt in range(max_retries):
            # Use a fresh name on each retry to avoid stale state
            if attempt > 0:
                self.container_name = f"tb2-{uuid.uuid4().hex[:12]}"
                logger.info("Retry %d/%d with new container name: %s",
                            attempt + 1, max_retries, self.container_name)

            subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)
            command = ["docker", "run", "-d", "--name", self.container_name,
                       "--platform", "linux/amd64"]
            if self.cpus:
                command += ["--cpus", str(self.cpus)]
            if self.memory_mb:
                command += ["--memory", f"{self.memory_mb}m"]
            for key, value in self.environment.items():
                command += ["-e", f"{key}={value}"]
            command += [self.image_name, "sh", "-c", "sleep infinity"]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                if attempt < max_retries - 1:
                    logger.warning("docker run failed (attempt %d/%d): %s — retrying in 2s",
                                   attempt + 1, max_retries, result.stderr.strip()[:200])
                    time.sleep(2)
                    continue
                raise RuntimeError(f"Failed to start container after {max_retries} attempts: {result.stderr}")

            # Verify the container is actually running (not immediately exited)
            time.sleep(0.5)
            check = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", self.container_name],
                capture_output=True, text=True,
            )
            if check.stdout.strip() == "true":
                self._running = True
                logger.info("Container '%s' started from image '%s'",
                            self.container_name, self.image_name)
                return self.container_name

            # Container exited immediately — get logs for diagnosis
            logs = subprocess.run(
                ["docker", "logs", self.container_name],
                capture_output=True, text=True,
            ).stdout[:300]
            subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)

            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning("Container exited immediately (attempt %d/%d), logs: %s — retrying in %ds",
                               attempt + 1, max_retries, logs.strip()[:150], wait)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Container exited immediately after {max_retries} attempts. "
                    f"Image: {self.image_name}. Last logs: {logs.strip()[:200]}"
                )

        raise RuntimeError(f"Failed to start container after {max_retries} attempts")

    def exec(self, command: str, workdir: str | None = None, timeout: int = 300) -> str:
        cmd = ["docker", "exec"]
        if workdir:
            cmd += ["-w", workdir]
        cmd += [self.container_name, "bash", "-c", command]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = (result.stdout or "") + (result.stderr or "")
        return output

    def copy_to(self, local_path: str, container_path: str) -> None:
        parent = str(Path(container_path).parent)
        self.exec(f"mkdir -p {parent}")
        result = subprocess.run(
            ["docker", "cp", local_path, f"{self.container_name}:{container_path}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to copy {local_path} -> {container_path}: {result.stderr}")

    def copy_directory_contents_to(self, local_dir: str, container_dir: str) -> None:
        """Copy a directory's contents without creating an extra nested level."""
        self.exec(f"mkdir -p {container_dir}")
        source = f"{Path(local_dir).resolve()}/."
        result = subprocess.run(
            ["docker", "cp", source, f"{self.container_name}:{container_dir}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to copy {source} -> {container_dir}: {result.stderr}")

    def run_tests(self, test_sh_path: str, timeout: int = 900,
                  verifier_env: dict | None = None) -> tuple[bool, str]:
        """Copy test.sh into the container, run it, and check reward.txt."""
        logger.info("Running test.sh (timeout=%ds)...", timeout)
        self.exec("mkdir -p /tests /logs/verifier")
        self.copy_to(test_sh_path, "/tests/test.sh")
        self.exec("chmod +x /tests/test.sh")
        env_prefix = " ".join(
            f"{key}={shlex.quote(str(value))}"
            for key, value in (verifier_env or {}).items()
        )
        command = f"{env_prefix} bash /tests/test.sh".strip()
        output = self.exec(command, timeout=timeout)
        reward = self.exec("cat /logs/verifier/reward.txt 2>/dev/null").strip()
        passed = reward == "1"
        logger.info("test.sh result: reward=%r, passed=%s", reward, passed)
        return passed, output

    NETWORK_ERROR_PATTERNS = [
        "Connection refused", "Connection reset", "Network is unreachable",
        "Temporary failure in name resolution", "Could not resolve host",
        "Failed to connect", "Failed to download", "Connection timed out",
        "No route to host", "The requested URL returned error",
        "502 Bad Gateway", "Could not get lock",
    ]

    def run_tests_with_retry(
        self, test_sh_path: str, timeout: int = 900, max_retries: int = 3,
        verifier_env: dict | None = None,
    ) -> tuple[bool, str]:
        """Run tests with retry on transient network errors."""
        for attempt in range(max_retries):
            passed, output = self.run_tests(
                test_sh_path, timeout=timeout, verifier_env=verifier_env
            )
            if passed:
                return passed, output
            output_lower = output.lower()
            is_transient = any(p.lower() in output_lower for p in self.NETWORK_ERROR_PATTERNS)
            if is_transient and attempt < max_retries - 1:
                wait = 2 ** attempt * 2
                logger.info("Transient error (attempt %d/%d), retrying in %ds...",
                            attempt + 1, max_retries, wait)
                time.sleep(wait)
                self.exec("rm -f /logs/verifier/reward.txt")
            else:
                return passed, output
        return passed, output

    def stop(self) -> None:
        if self._running:
            subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)
            self._running = False
            logger.info("Container '%s' stopped and removed.", self.container_name)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def pull_image(image_name: str) -> bool:
    """Pull a Docker image if not already present locally."""
    check = subprocess.run(["docker", "image", "inspect", image_name], capture_output=True)
    if check.returncode == 0:
        return True
    logger.info("Pulling image %s...", image_name)
    result = subprocess.run(
        ["docker", "pull", image_name],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        logger.error("Failed to pull %s: %s", image_name, result.stderr.strip())
        return False
    return True
