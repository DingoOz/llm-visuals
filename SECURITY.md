# Security

## Reporting

Report a vulnerability in private:

https://github.com/DingoOz/llm-visuals/security/advisories/new

Please do not open a public issue, pull request, or discussion for an unfixed vulnerability. A public report includes the sample you would otherwise paste, so do not put a proof of concept there either.

Include the version (`llm-visuals --version` or the release tag), the operating system, what an attacker can do, and the smallest steps that show it.

## In scope

- The `llm-visuals` binary and the release archives published from this repository.
- An update check that downloads one of those archives and replaces the binary. The trust boundary is the GitHub release and the sha256 published next to it.
- API keys read from a server command line or from `--api-key-file`. The token is not written to saved settings or the log database; a path that leaks it is in scope.
- Process and GPU inspection on the machine where the dashboard is running.

## Out of scope

- llama.cpp, vLLM, SGLang, Ollama, LM Studio, the GPU driver, or the model weights.
- A terminal emulator, a shell, or another program on the same machine.
- A dashboard that shows wrong numbers with no security impact. That is a normal bug report.
