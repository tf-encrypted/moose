certs:
	rm -rf certs/
	certstrap --depot-path certs init --common-name ca --passphrase ""
	certstrap --depot-path certs request-cert --domain worker0 --passphrase ""
	certstrap --depot-path certs sign --CA ca worker0
	certstrap --depot-path certs request-cert --domain worker1 --passphrase ""
	certstrap --depot-path certs sign --CA ca worker1
	certstrap --depot-path certs request-cert --domain worker2 --passphrase ""
	certstrap --depot-path certs sign --CA ca worker2
	certstrap --depot-path certs request-cert --domain worker3 --passphrase ""
	certstrap --depot-path certs sign --CA ca worker3
	certstrap --depot-path certs request-cert --domain choreographer --passphrase ""
	certstrap --depot-path certs sign --CA ca choreographer

up:
	docker-compose up --build --remove-orphans

down:
	docker-compose down --remove-orphans

run:
	docker-compose exec choreographer \
		sh -c "python3 choreographer.py --verbose --runtime remote"

.PHONY: certs up down run
