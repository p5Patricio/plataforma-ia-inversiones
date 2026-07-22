# Delta for Professional Operations

## ADDED Requirements

### Requirement: SDD Change Tracking

The repository MUST track professional improvement work through OpenSpec artifacts before implementation.

#### Scenario: Active change is inspectable

- GIVEN the professional improvements foundation is in development
- WHEN a reviewer opens `openspec/changes/professional-improvements-foundation/`
- THEN proposal, specs, design, tasks, and state artifacts exist
- AND they describe scope, success criteria, and verification.

### Requirement: Secret-Free CI

The repository MUST run routine pull request checks without requiring Supabase production secrets.

#### Scenario: CI runs on pull request

- GIVEN code is pushed or proposed in a pull request
- WHEN the CI workflow runs
- THEN Python tests execute with test/demo-safe settings
- AND frontend lint/build checks execute from `ui/`
- AND no production operational job is triggered by CI.

### Requirement: Maintainer Guidance

The repository MUST document contribution, security, and release-history expectations.

#### Scenario: Contributor verifies a change

- GIVEN a contributor changes backend, frontend, or docs
- WHEN they read repository guidance
- THEN they can find the relevant verification commands
- AND they understand that `.env` and private Supabase keys MUST NOT be committed.
