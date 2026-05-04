# FOR AGENT

------

## Commands

All terminal commands you are allowed to run:

* Check

```bash
make rs-check
```

* Format

```bash
make rs-fmt
```

* Clippy

```bash
make rs-clippy
```

* Lint (Python)

```bash
make python-lint
```

**Always ask** before running any other command unless you are explicitly told to do so.

------

## Project Structure

```
.
├── AGENTS.md
├── Cargo.lock
├── Cargo.toml
├── clippy.toml
├── README.md
├── Makefile
├── flow.mermaid
├── rustfmt.toml
├── pyproject.toml
├── yolo-export.py
├── remote-control.py
├── target
├── assets
│   └── models
│       └── yolo
├── .cache
│   └── models
├── .venv
├── src -> core logic
│       ├── com -> communication functionalities (Bluetooth, buffer pool)
│       ├── control -> object tracker and velocity regulator
│       ├── cv -> Computer Vision functionalities: GStreamer pipeline, image processing, bounding boxes
│       ├── utils -> general application-wide utilities
│       ├── ml -> Machine Learning components
│       ├── lib.rs
│       └── main.rs
```

------

## Operational Constraints

### Scope of Work

- You may only work inside the project’s root directory and its subdirectories.
- You must not access, read, or modify files outside the repository.
- You must not interact with the host system beyond what is required to build, test, or lint this project.

### File System Rules

- You may only add new files or modify existing files within the repository.
- You must not delete files unless explicitly instructed.
- You must not rename or move files unless explicitly instructed.
- You must not add new files unless explicitly instructed.
- You must not modify generated files directly nor modify git-ignored files.

### Command Execution Rules

- You must never run commands with `sudo`.
- You must not attempt to elevate privileges.
- You may only run safe, project-local commands.
- You must not install global system packages.
- You must not modify system configuration.

### Dependency Management

- You may add dependencies only if necessary for the requested change.
- **Always ask** before adding a new dependency.
- You must avoid large or unnecessary third-party libraries.
- You must not remove, downgrade, or upgrade dependencies unless explicitly instructed.

### Network and External Access

- You must not make arbitrary external network calls.
- You must not introduce code that sends data to unknown external services.
- You must not embed credentials, tokens, or secrets.

### Code Change Principles

- You must limit changes strictly to what is required.
- You must not modify unrelated code in any way.
- You should not add new traits, functions, variables, or any other data structures unless explicitly instructed.
**Always ask** before introducing a new component.
- You must ensure the project builds and tests pass after modifications.

### Security and Safety

- You must not introduce hardcoded secrets.
- You must avoid unsafe patterns.

### Documentation and Transparency

- You must update documentation if behavior changes.
- You must keep documentation informative and concise.
- You must make sure that added documentation follows the format used within the repository. Reference code
components within the documentation (if applicable), e.g. `[ort::value::Tensor]`.

If any requested task requires violating these rules, you must refuse the task and explain why.

------

## Coding Guidelines

1. Follow general Rust conventions and the coding style embraced within the repository.
2. Use short and meaningful names for naming components. Traits should describe behavior.
3. **Always** handle errors explicitly. **Never** use panics.
4. **Never** use unsafe features.
5. Log relevant information with a proper level using the logging format embraced within the repository. 
**Never** log sensitive information.
6. **Always** add unit tests for newly added functionalities or modify existing ones if the behavior they cover
has been modified. Tests should be added within the `mod tests` at the bottom of the file containing tested
functionality. **Always** add `test_` prefix to the test function name.
