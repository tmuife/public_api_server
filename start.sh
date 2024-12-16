source /home/ubuntu/project/public_api_server/Florence2_server/.venv/bin/activate
cd /home/ubuntu/project/public_api_server/Florence2_server
python main.py >> run.log 2>&1 &

deactivate

source /home/ubuntu/project/public_api_server/embedding_server/.venv/bin/activate
cd /home/ubuntu/project/public_api_server/embedding_server
python main.py >> run.log 2>&1 &


