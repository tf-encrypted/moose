
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

run:
	docker run \
		--tty \
        --network python-functions_default \
		--volume $(realpath ../../moose):/moose \
		--volume $(realpath .):/examples/python-functions \
		${MAIN_IMAGE} \
		sh -c "python3 /examples/python-functions/main.py --runtime remote --verbose --cluster-spec /examples/python-functions/cluster-spec.yaml"

test-localhost:
	docker run \
		--tty \
		--volume $(realpath ../../moose):/moose \
		--volume $(realpath .):/examples/python-functions \
		${MAIN_IMAGE} \
		sh -c "python3 /examples/python-functions/main.py --runtime test --verbose "



.PHONY: certs run test-localhost
