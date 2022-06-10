# Examples

## TLS support

The (insecure) certificates used for the examples were generated as follows using [certstrap](https://github.com/square/certstrap):

```sh
certstrap --depot-path certs init --common-name ca --passphrase ""

certstrap --depot-path certs request-cert --common-name choreographer --domain localhost --passphrase ""
certstrap --depot-path certs sign --CA ca choreographer

certstrap --depot-path certs request-cert --common-name localhost:50000 --domain localhost --passphrase ""
certstrap --depot-path certs sign --CA ca localhost_50000

certstrap --depot-path certs request-cert --common-name localhost:50001 --domain localhost --passphrase ""
certstrap --depot-path certs sign --CA ca localhost_50001

certstrap --depot-path certs request-cert --common-name localhost:50002 --domain localhost --passphrase ""
certstrap --depot-path certs sign --CA ca localhost_50002
```
