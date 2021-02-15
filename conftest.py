import os

from hypothesis import settings

settings.register_profile("ci", max_examples=1)
settings.register_profile("test-standard", max_examples=10)
settings.register_profile("test-long", max_examples=100)


settings.load_profile(
    os.getenv(u"HYPOTHESIS_PROFILE", "test-standard")
)  # change the default here
