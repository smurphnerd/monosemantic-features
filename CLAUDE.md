# Project Guidelines

## Testing Philosophy

- **Unit test all functions** - Every function should have unit tests covering its behavior
- **No arbitrary metric goals in tests** - Avoid tests that check for specific numeric thresholds like "should achieve within 2x of Welch bound"
- **Skip integration tests with soft goals** - The researcher will run experiments to validate research ideas work end-to-end
- **Test behavior, not benchmarks** - Test that functions return correct types, shapes, and satisfy mathematical invariants (e.g., unit norm, positive values where expected)

## Worktrees

Worktree directory: `.worktrees/` (gitignored)
