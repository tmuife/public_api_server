from __future__ import annotations

import unittest

from app.config import KafkaAuthSettings, KafkaSettings
from app.services.kafka_service import KafkaService


class FakeTopicMetadata:
    error = None


class FakeClusterMetadata:
    def __init__(self, topic: str):
        self.topics = {topic: FakeTopicMetadata()}


class FakeAdminClient:
    configs: list[dict] = []

    def __init__(self, config: dict):
        self.config = config
        self.configs.append(config)

    def list_topics(self, topic: str, timeout: int):
        _ = timeout
        return FakeClusterMetadata(topic)


class FakeConsumer:
    configs: list[dict] = []
    subscriptions: list[list[str]] = []

    def __init__(self, config: dict):
        self.config = config
        self.configs.append(config)

    def subscribe(self, topics: list[str]) -> None:
        self.subscriptions.append(topics)


class FakeProducer:
    configs: list[dict] = []

    def __init__(self, config: dict):
        self.config = config
        self.configs.append(config)


class FakeNewTopic:
    def __init__(self, topic: str, num_partitions: int, replication_factor: int):
        self.topic = topic
        self.num_partitions = num_partitions
        self.replication_factor = replication_factor


class KafkaServiceCredentialTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeAdminClient.configs = []
        FakeConsumer.configs = []
        FakeConsumer.subscriptions = []
        FakeProducer.configs = []

    @staticmethod
    def _settings() -> KafkaSettings:
        return KafkaSettings(
            bootstrap_servers="broker:9092",
            security_protocol="SASL_PLAINTEXT",
            sasl_mechanism="SCRAM-SHA-512",
            ssl_ca_file=None,
            produce_timeout_seconds=1.0,
            wait_first_frame_seconds=1.0,
            idle_timeout_seconds=1.0,
            max_process_seconds=1.0,
            topic_partitions=1,
            topic_replication_factor=1,
            consumer_auth=KafkaAuthSettings(
                sasl_username="consumer-user",
                sasl_password="consumer-pass",
            ),
            producer_auth=KafkaAuthSettings(
                sasl_username="producer-user",
                sasl_password="producer-pass",
            ),
        )

    def test_kafka_clients_use_purpose_specific_credentials(self) -> None:
        service = KafkaService(self._settings())
        original_loader = KafkaService._load_kafka_modules
        KafkaService._load_kafka_modules = staticmethod(
            lambda: (FakeConsumer, object, FakeProducer, FakeAdminClient, FakeNewTopic)
        )

        try:
            service.ensure_topic("job_1776945123456_ab12cd34_output")
            service.create_frame_consumer(topic="job_1776945123456_ab12cd34_input", group_id="debug-run-001")
            service.create_frame_producer()
        finally:
            KafkaService._load_kafka_modules = original_loader

        self.assertEqual(FakeAdminClient.configs[0]["sasl.username"], "producer-user")
        self.assertEqual(FakeAdminClient.configs[0]["sasl.password"], "producer-pass")
        self.assertEqual(FakeConsumer.configs[0]["sasl.username"], "consumer-user")
        self.assertEqual(FakeConsumer.configs[0]["sasl.password"], "consumer-pass")
        self.assertEqual(FakeConsumer.configs[0]["group.id"], "debug-run-001")
        self.assertEqual(FakeProducer.configs[0]["sasl.username"], "producer-user")
        self.assertEqual(FakeProducer.configs[0]["sasl.password"], "producer-pass")


if __name__ == "__main__":
    unittest.main()
