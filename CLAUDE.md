## Design Principles

Apply these when writing or reviewing code (from *A Philosophy of Software Design*, Ousterhout):

1. **Complexity is incremental — sweat the small stuff.** No single change makes a system complex; the accumulation does.
2. **Working code isn't enough.** Passing tests is the floor, not the goal.
3. **Make continual small investments to improve system design.** Spend a little extra now to avoid tactical debt.
4. **Modules should be deep.** Simple interface, substantial functionality behind it.
5. **Design interfaces to make the most common usage as simple as possible.** Optimize for the caller's common case.
6. **A simple interface matters more than a simple implementation.** Accept internal complexity to spare every caller from it.
7. **General-purpose modules are deeper.** Prefer a somewhat generic API over one narrowly fit to today's caller.
8. **Separate general-purpose and special-purpose code.** Don't let special cases leak into general layers.
9. **Different layers should have different abstractions.** If a layer just forwards to the next, it's pass-through cruft — remove it.
10. **Pull complexity downward.** Better for the module author to suffer than every user of the module.
11. **Define errors (and special cases) out of existence.** Redesign the API so the error condition can't occur.
12. **Design it twice.** Sketch at least two approaches before committing to one.
13. **Comments should describe things that are not obvious from the code.** Capture intent, invariants, and rationale — not restatements.
14. **Software should be designed for ease of reading, not ease of writing.** Reading happens far more often.
15. **The increments of software development should be abstractions, not features.** When adding a feature, find the clean abstraction it implies.

