import densenet121
import inceptionv3
import resnet152v2
import vgg16

import utilities.evaluate as evaluate


def main():
    # Call the main transfer learning for all 12 variation models
    # print(resnet152v2.run())
    # print(densenet121.run())
    print(inceptionv3.run())
    # print(vgg16.run())

    # Export the variation models for use in offensive testing
    print("We Gotta Do This Soons!")

if __name__ == "__main__":
    main()
