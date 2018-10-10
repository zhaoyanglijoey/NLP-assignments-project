import argparse
import string
import random


def to_cipher(key, plaintext):
    ciphertext = []
    for c in plaintext:
        if c not in string.ascii_letters:
            continue
        if c not in string.ascii_lowercase:
            c = c.lower()
        ciphertext.append(random.choice(key[c]))

    return ''.join(ciphertext)


def read_key_file(keyfile):
    key = {}  # plaintext -> ciphertext
    with open(keyfile, 'r') as f:
        for line in f:
            plain_cipher = line.strip().split()
            if len(plain_cipher) < 2:
                continue
            plain = plain_cipher[0]
            cipher = plain_cipher[1:]
            key[plain] = cipher
    return key


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-k', '--key_file', default='ciphergenerator/key.txt', help='key file')
    arg_parser.add_argument('-p', '--plaintext_file', default='ciphergenerator/plaintext.txt', help='plaintext file')
    arg_parser.add_argument('-c', '--ciphertext_file', default='ciphergenerator/ciphertext.txt', help='ciphertext file')
    args = arg_parser.parse_args()
    key = read_key_file(args.key_file)

    with open(args.plaintext_file, 'r') as f:
        plaintext = f.read()

    with open(args.ciphertext_file, 'w') as f:
        f.write(to_cipher(key, plaintext))
