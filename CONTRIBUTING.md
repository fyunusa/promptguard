# Contributing to PromptGuard

We welcome contributions from the community! Here's how to get involved.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/promptguard.git`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Install dev dependencies: `pip install -e ".[all]" && pip install -r requirements-dev.txt`
5. Make your changes
6. Write tests for new functionality
7. Run tests: `pytest tests/ -v`
8. Commit with clear messages: `git commit -m "feat: add new feature"`
9. Push to your fork
10. Create a Pull Request

## Code Style

- Follow [Black](https://github.com/psf/black) for formatting
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Type hints are required for all public APIs
- Docstrings for all public functions and classes

## Testing

- Aim for >90% code coverage
- Write unit tests for new functionality
- Include integration tests for provider implementations
- Use `pytest` and `pytest-asyncio` for async tests

## PR Guidelines

- Keep PRs focused and reasonably sized
- Include tests for new features
- Update documentation as needed
- Clear description of changes and rationale
- Link related issues

## Release Process

Releases follow [Semantic Versioning](https://semver.org/):
- Major: Breaking changes
- Minor: New features
- Patch: Bug fixes

## Questions?

Open an issue or discussion for questions and feature requests.
