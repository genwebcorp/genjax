# The menagerie of `GenerativeFunction`
Generative functions are probabilistic building blocks. They allow you to express complex probability distributions, and automate several operations on them. GenJAX exports a standard library of generative functions, and this page catalogues them and their usage.
## The venerable & reliable `Distribution`

To start, distributions are generative functions.

::: genjax.Distribution
    options:
        show_root_heading: true
        members:
          - random_weighted
          - estimate_logpdf

Distributions intentionally expose a permissive interface ([`random_weighted`](generative_functions.md#genjax.Distribution.random_weighted) and [`estimate_logpdf`](generative_functions.md#genjax.Distribution.estimate_logpdf) which doesn't assume _exact_ density evaluation. [`genjax.ExactDensity`](generative_functions.md#genjax.ExactDensity) is a more restrictive interface, which assumes exact density evaluation.

::: genjax.ExactDensity
    options:
        show_root_heading: true
        members:
          - random_weighted
          - estimate_logpdf

## `StaticGenerativeFunction`: a programmatic language

For any serious work, you'll want a way to combine generative functions together, mixing deterministic functions with sampling. `StaticGenerativeFunction` is a way to do that: it supports the use of a JAX compatible subset of Python to author generative functions. It also supports the ability _to invoke_ other generative functions: instances of this type (and any other type of generative function) can then be used in larger generative programs.

::: genjax.StaticGenerativeFunction
    options:
        show_root_heading: true
        members:
        - source
        - simulate
        - assess
        - update

## Combinators: patterns for composition

::: genjax.VmapCombinator
    options:
        show_root_heading: true
        members:
        - gen_fn
        - in_axes
        - simulate
        - assess
        - update

::: genjax.ScanCombinator
    options:
        show_root_heading: true

::: genjax.SwitchCombinator
    options:
        show_root_heading: true

::: genjax.MaskCombinator
    options:
        show_root_heading: true

## Derived combinators


::: genjax.ComposeCombinator
    options:
        show_root_heading: true

::: genjax.repeat_combinator
    options:
        show_root_heading: true

::: genjax.cond_combinator
    options:
        show_root_heading: true

::: genjax.mixture_combinator
    options:
        show_root_heading: true

::: genjax.address_bijection_combinator
    options:
        show_root_heading: true
