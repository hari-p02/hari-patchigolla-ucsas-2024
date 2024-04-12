# Soccer Analytics with Tensorflow!

To set up your ec2 instance you can use these commands:

```
sudo yum install python3-pip
pip3 install pymilvus
mkdir milvus
cd milvus
wget https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo yum install docker
sudo systemctl start docker
sudo docker-compose up -d
```

Open port 5000, 19530, and 8000

```
sudo docker ps -a
run code in milvus.ipynb
sudo docker run -d -p 8000:3000 -e HOST_URL=http://EC2_PUBLIC_IP:8000 -e MILVUS_URL=http://host.docker.internal:19530 zilliz/attu:latest
http://EC2_PUBLIC_IP:19530
```
```
scp -i ~/path_to_pem/ucsas.pem playerstoinf.json app.py embds.npy embedding_model.h5 ec2-user@EC2_PUBLIC_IP:/home/ec2-user
pip3 install flask
pip3 install flask-cors
pip3 install tensorflow-cpu
pip3 install category-encoders
```
