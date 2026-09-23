# Motion public source deploy contract

## Scope and source identity

This repository publishes the Motion detector and tracker source, plus a
compatibility command for analysis moved to PyMicroglia. It is not a Python
distribution. The deployed source identity is the verified commit on GitHub
`Jay2owe/motion-microglia-detection` `main`.

## Destination and order

Publish the compatibility command to GitHub `main` after PyMicroglia 0.3.0 is
available on PyPI. There is no other documented distribution channel or release
artifact for this repository. Keep `code/` byte-identical to the preceding
public `main` commit; the independent Motion tracking work stays in its working
checkout. The historical command depends on `PyMicroglia[states]>=0.3,<0.4`.

## Gates

- Check the compatibility command against an installation of the published
  PyMicroglia release: `python -m analysis doctor`, `modules`, and a
  fixture-backed analysis command.
- Check that `code/` contains no dependency on `analysis` and has the same
  tree hash as the preceding public commit.
- Run the public push guard before committing and pushing. Verify the GitHub
  `main` commit and the command against the published PyMicroglia wheel.
