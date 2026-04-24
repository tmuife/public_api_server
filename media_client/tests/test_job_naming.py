from __future__ import annotations

import unittest

from app.utils.job import (
    JOB_NAME_PATTERN,
    derive_input_topic,
    derive_job_name_from_topic,
    generate_job_name,
)


class JobNamingTests(unittest.TestCase):
    def test_generate_job_name_matches_required_pattern(self) -> None:
        job_name = generate_job_name()
        self.assertRegex(job_name, JOB_NAME_PATTERN)

    def test_input_topic_derives_from_job_name(self) -> None:
        job_name = "job_1776945123456_ab12cd34"
        self.assertEqual(derive_input_topic(job_name), "job_1776945123456_ab12cd34_input")

    def test_job_name_can_be_resolved_from_output_topic(self) -> None:
        topic = "job_1776945123456_ab12cd34_output"
        self.assertEqual(derive_job_name_from_topic(topic), "job_1776945123456_ab12cd34")


if __name__ == "__main__":
    unittest.main()
