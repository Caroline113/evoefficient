"""Scope the container proxy to the Harbor delegate installation."""
import shlex

from agent_evolve.agents.terminal.harbor_adapter import AEvolveTerminalHarborAgent


class SetupProxyHarborAgent(AEvolveTerminalHarborAgent):
    def __init__(self, *args, setup_proxy, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_proxy = setup_proxy

    async def setup(self, environment):
        keys = ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY')
        proxy_env = dict.fromkeys(keys, self.setup_proxy)
        # Keep these APT settings local to this setup. npm's recommended
        # desktop packages add hundreds of downloads on Debian.
        apt_settings = 'Acquire::Retries "4";\nAPT::Install-Recommends "false";\n'
        command = f'cfg=$(mktemp) && printf %s {shlex.quote(apt_settings)} > "$cfg" && printf %s "$cfg"'
        result = await environment.exec(command)
        if result.return_code != 0 or not (result.stdout or '').strip():
            raise RuntimeError('Could not create temporary APT config for Harbor agent setup')
        apt_config = result.stdout.strip()
        try:
            with environment.scoped_exec_env({**proxy_env, 'APT_CONFIG': apt_config}):
                await super().setup(environment)
        finally:
            await environment.exec(f'rm -f -- {shlex.quote(apt_config)}')
