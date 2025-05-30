import random
import unittest

import torch
import torch.nn as nn

import src.utils_torch as utils_torch


class TestSeedEverything(unittest.TestCase):

    def test_random_sampling(self):
        utils_torch.seed_everything(42)
        sample1 = random.sample(range(100), 10)

        utils_torch.seed_everything(42)
        sample2 = random.sample(range(100), 10)

        self.assertEqual(sample1, sample2)

    def test_random_sampling2(self):
        utils_torch.seed_everything(42)
        sample1 = random.sample(range(100), 10)

        utils_torch.seed_everything(42)
        _ = random.randint(0, 100)
        sample2 = random.sample(range(100), 10)

        self.assertNotEqual(sample1, sample2)


class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class TestModelExact(unittest.TestCase):

    def test_model_exact_1(self):
        utils_torch.seed_everything(123)
        model_1 = SimpleModel()
        utils_torch.seed_everything(123)
        model_2 = SimpleModel()

        self.assertTrue(utils_torch.is_model_pair_exact(model_1, model_2))

    def test_model_exact_2(self):
        utils_torch.seed_everything(123)
        model_1 = SimpleModel()
        utils_torch.seed_everything(124)
        model_2 = SimpleModel()

        self.assertFalse(utils_torch.is_model_pair_exact(model_1, model_2))


class TestModelDevice(unittest.TestCase):

    def test_cpu(self):
        model = SimpleModel()
        device = utils_torch.get_model_device(model)
        self.assertTrue(device == torch.device("cpu"))


class TestModelFreezing(unittest.TestCase):

    def test_freeze_model(self):
        model = SimpleModel()

        # Ensure the model is initially unfrozen
        self.assertFalse(utils_torch.is_all_frozen(model))

        # Freeze the model
        utils_torch.freeze_model(model)

        # Check if the model is fully frozen
        self.assertTrue(utils_torch.is_all_frozen(model))


class TestModelUnFreezing(unittest.TestCase):

    def test_unfreeze_model(self):

        model = SimpleModel()

        # Ensure the model is initially unfrozen
        self.assertFalse(utils_torch.is_all_frozen(model))

        # Freeze the model
        utils_torch.freeze_model(model)

        # Check if the model is fully frozen
        self.assertTrue(utils_torch.is_all_frozen(model))

        utils_torch.unfreeze_model(model)

        # Check if the model is fully unfrozen
        self.assertFalse(utils_torch.is_all_frozen(model))
        self.assertFalse(utils_torch.is_any_frozen(model))


class TestGradRequiredLoadAndGet(unittest.TestCase):

    @staticmethod
    def get_model():
        model = SimpleModel()

        for param in model.parameters():
            param.requires_grad_(True)

        for param in model.fc1.parameters():
            param.requires_grad_(False)

        return model

    def test_get_grad_required_state(self):
        model = self.get_model()

        self.assertTrue(utils_torch.is_any_frozen(model))
        self.assertFalse(utils_torch.is_all_frozen(model))

        state = utils_torch.get_grad_required_state(model)

        # Further checks can be added here based on expected state contents
        self.assertTrue("fc1.weight" not in state)
        self.assertTrue("fc1.bias" not in state)
        self.assertTrue("fc2.weight" in state)
        self.assertTrue("fc2.bias" in state)

    def test_load_grad_required_state(self):

        utils_torch.seed_everything(123)
        model_1 = self.get_model()
        utils_torch.seed_everything(124)
        model_2 = self.get_model()

        self.assertFalse(
            utils_torch.is_model_pair_exact(model_1.fc1, model_2.fc1)
        )
        self.assertFalse(
            utils_torch.is_model_pair_exact(model_1.fc2, model_2.fc2)
        )

        states = utils_torch.get_grad_required_state(model_1)

        model_2 = utils_torch.load_grad_required_state(
            model_2, states, verbose=False
        )
        self.assertFalse(
            utils_torch.is_model_pair_exact(model_1.fc1, model_2.fc1)
        )
        self.assertTrue(
            utils_torch.is_model_pair_exact(model_1.fc2, model_2.fc2)
        )


class TestSetGradRequiredLayerTrain(unittest.TestCase):

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            utils_torch.set_grad_required_layer_train("not a module")

    def test_all_layers_frozen(self):
        model = SimpleModel()
        utils_torch.freeze_model(model)

        utils_torch.set_grad_required_layer_train(model)
        # Model should be in eval mode
        self.assertFalse(model.training)

    def test_recursive_behavior(self):
        model = SimpleModel()
        utils_torch.freeze_model(model)

        utils_torch.set_grad_required_layer_train(model)
        for child in model.children():
            # Each child should be in eval mode
            self.assertFalse(child.training)

    def test_partial_layers_frozen(self):
        model = SimpleModel()
        utils_torch.freeze_model(model.fc1)

        utils_torch.set_grad_required_layer_train(model)
        self.assertFalse(model.fc1.training)
        self.assertTrue(model.fc2.training)

    def test_from_eval_to_train1(self):
        model = SimpleModel()
        model.eval()

        utils_torch.set_grad_required_layer_train(model)

        self.assertTrue(model.training)

    def test_from_eval_to_train2(self):
        model = SimpleModel()
        utils_torch.freeze_model(model.fc1)
        model.eval()

        utils_torch.set_grad_required_layer_train(model)

        self.assertTrue(model.training)
        self.assertFalse(model.fc1.training)
        self.assertTrue(model.fc2.training)

    def test_from_eval_to_train3(self):
        model = SimpleModel()
        utils_torch.freeze_model(model)

        model.eval()
        utils_torch.set_grad_required_layer_train(model)

        self.assertFalse(model.training)
        self.assertFalse(model.fc1.training)
        self.assertFalse(model.fc2.training)


if __name__ == "__main__":
    unittest.main()
