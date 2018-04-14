class Batch:
    def __init__(self):
        self.images = None
        self.labels = None
        self.batch_size = None

        self.batch_image = None
        self.batch_label = None

        self.size_of_data = None
        self.left = 0
        self.current_batch = 0
        self.previous_end = 0
        self.offset = 0
        self.batch_images = None
        self.batch_labels = None

    def set_data(self, images, labels, batch_size):

        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.size_of_data = len(self.images)
        self.left = self.size_of_data

    def next_batch(self):
        self.offset = (self.current_batch * self.batch_size) % (self.size_of_data - self.batch_size)
        if self.left > self.batch_size:
                self.left = self.left - self.batch_size
                self.previous_end = self.offset + self.batch_size
                self.batch_images = self.images[self.offset: (self.offset + self.batch_size)]
                self.batch_labels = self.labels[self.offset: (self.offset + self.batch_size)]
        else:
                self.batch_images = self.images[self.previous_end: self.size_of_data]
                self.batch_labels = self.labels[self.previous_end: self.size_of_data]

        self.current_batch = self.current_batch + 1

        return self.batch_images, self.batch_labels

class Prepare_Batch:
    def __init__(self):
        ''

    def prepare(self, images, labels, batch_size):
        bt = Batch()
        bt.set_data(images, labels, batch_size)
        # print("Batch prepared with details images", str(len(bt.images)), "labels", str(len(bt.labels)))
        return bt