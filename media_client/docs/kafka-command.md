docker exec -it kafka kafka-topics \
  --bootstrap-server localhost:9092 \
  --create \
  --topic test-topic \
  --partitions 3 \
  --replication-factor 1


# 查看所有 topic
docker exec -it kafka kafka-topics \
  --bootstrap-server localhost:9092 --list

# 查看 topic 详情
docker exec -it kafka kafka-topics \
  --bootstrap-server localhost:9092 \
  --describe --topic test-topic

# 删除 topic
docker exec -it kafka kafka-topics \
  --bootstrap-server localhost:9092 \
  --delete --topic test-topic
