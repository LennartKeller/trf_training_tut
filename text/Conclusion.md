# Conclusion

Due to their different scopes labeling one of the presented frameworks as the "best one" would be unfair.
Obviously, the built-in of the `Trainer` of the `transformers` library is optimally aligned with the rest of the library, making it the best choice when training standard models.
Even in cases like ours, where a standard model is trained with a custom loss, the `Trainer` requires few adaptions.
However, from a conceptual point of view, PyTorch Lightning approach is far more sustainable since its API forces to structure the code into mostly self-contained models.
While requiring more manual implementation, this approach leads to better code quality and makes models and datasets more reusable.
T
