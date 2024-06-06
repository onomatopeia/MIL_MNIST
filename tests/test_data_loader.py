from data_loader import MnistBags


def test_mnist_bags():
    num_bags = 10
    target_number = 4
    bags = MnistBags(target_number=target_number, num_bag=num_bags, seed=7)
    assert len(bags.labels) == num_bags
    assert len(bags.bags) == num_bags
    assert len(bags) == num_bags
    for bag, [bag_label, images_labels] in bags:
        assert len(bag) == len(images_labels)
        assert bag_label in [0, 1]
        if bag_label == 0:
            assert True not in images_labels
        else:
            assert True in images_labels
