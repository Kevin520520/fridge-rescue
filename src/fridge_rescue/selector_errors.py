"""Typed failures at the recommendation-selector boundary.

The workflow may retry malformed model output once, but must not retry service
failures such as authentication, quota, network, or provider refusals.
"""


class SelectorError(RuntimeError):
    """Base class for failures raised by a recommendation selector."""


class SelectorServiceError(SelectorError):
    """The external recommendation service could not provide a draft."""


class SelectorOutputError(SelectorError):
    """The service responded, but its draft was malformed or out of bounds."""
