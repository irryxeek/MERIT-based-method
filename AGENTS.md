# Repository Guidelines

## Project Structure & Module Organization

`MERIT/` currently has no application code, dependency manifests, tests, or Git metadata. Use the following proposed layout as the project grows; create directories only when needed:

- `src/`: application or library code, grouped by feature or subsystem.
- `tests/`: automated tests mirroring the source structure.
- `assets/`: static resources and small fixtures.
- `docs/`: architecture notes and contributor documentation.

Add a root `README.md` describing the project's purpose and setup when the first implementation lands.

## Build, Test, and Development Commands

No install, build, test, or local-run commands are configured yet. Run future project commands from the repository root. When introducing tooling, document exact setup, development, lint, test, and build commands in `README.md` and update this guide. Declare dependencies in the chosen ecosystem's manifest and record required runtime versions. Do not assume globally installed tools.

## Coding Style & Naming Conventions

Follow the chosen language's standard formatter and commit its configuration. Use consistent indentation within each file and avoid unrelated formatting changes. Prefer descriptive names tied to behavior; keep modules focused on one responsibility. Match filenames to their exported module or feature, and document language-specific naming and indentation rules once the stack is selected.

## Testing Guidelines

No testing framework or coverage threshold is established. Configure an appropriate framework with the first executable code and document a reproducible test command. Keep tests deterministic and independent of private services where possible. Use the framework's discovery convention, such as `test_*.py` for pytest or `*.test.ts` for a TypeScript test runner. Include focused regression tests with behavior fixes.

## Commit & Pull Request Guidelines

No Git history is available to establish existing conventions. Use concise, imperative commit subjects, for example, `Add input validation`. Keep each commit focused on one change. Pull requests should explain the problem, summarize the resulting behavior, link relevant issues, and list validation commands and results. Include screenshots for visible interface changes and identify any setup or migration steps reviewers need.
