
MAIN_IMAGE?=tfencrypted/runtime-base-worker:latest

certs:
	rm -rf certs/
	certstrap --depot-path certs init --common-name ca --passphrase ""
	certstrap --depot-path certs request-cert --domain inputter0 --passphrase ""
	certstrap --depot-path certs sign --CA ca inputter0
	certstrap --depot-path certs request-cert --domain inputter1 --passphrase ""
	certstrap --depot-path certs sign --CA ca inputter1
	certstrap --depot-path certs request-cert --domain aggregator --passphrase ""
	certstrap --depot-path certs sign --CA ca aggregator
	certstrap --depot-path certs request-cert --domain outputter --passphrase ""
	certstrap --depot-path certs sign --CA ca outputter
	certstrap --depot-path certs request-cert --domain coordinator --passphrase ""
	certstrap --depot-path certs sign --CA ca coordinator

up:
	docker-compose up --build

down:
	docker-compose down --remove-orphans

run:
	docker-compose exec coordinator sh -c "python3 /example/main.py --verbose --runtime remote"

.PHONY: certs up down run
