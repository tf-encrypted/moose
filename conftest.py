import os
from datetime import timedelta

from hypothesis import settings

settings.register_profile(
    "ci-long", max_examples=100, deadline=timedelta(milliseconds=2000)
)
settings.register_profile("ci", max_examples=1, deadline=timedelta(milliseconds=2000))


settings.load_profile(os.getenv(u"HYPOTHESIS_PROFILE", "default"))
