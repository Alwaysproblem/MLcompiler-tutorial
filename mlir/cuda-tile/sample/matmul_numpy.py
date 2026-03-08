import numpy as np


def main() -> None:
    # Toy: var a = [[1, 2, 3, 9], [4, 5, 6, 10]]
    a = np.array([[1, 2, 3, 9], [4, 5, 6, 10]], dtype=np.int64)

    # Toy: var b<2, 4> = [11, 12, 13, 14, 15, 16, 17, 18]
    b = np.array([11, 12, 13, 14, 15, 16, 17, 18], dtype=np.int64).reshape(2, 4)

    # Toy: print(a * b + b)
    print((a * b + b))

    # Toy: print(matmul(a, transpose(b)))
    print(np.matmul(a, b.T))

    # Toy: var c<2, 4> = [[7, 8, 9, 13], [10, 11, 12, 14]]
    # Toy: var d<2, 4> = [[7, 8, 9, 13], [10, 11, 12, 14]]
    c = np.array([[7, 8, 9, 13], [10, 11, 12, 14]], dtype=np.int64)
    d = np.array([[7, 8, 9, 13], [10, 11, 12, 14]], dtype=np.int64)

    # Toy: print(a * c + b * d)
    print((a * c + b * d))


if __name__ == "__main__":
    main()
