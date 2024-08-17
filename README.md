# Hybrid VAE-GAN Architecture for Artistic Style Transfer

## Architecture
<p align="center">
  <img src="/home/hemanth/Pictures/Screenshots/Screenshot2024-08-05_19:47:40.png">
</p>

This package provides reference implementation of the `Hybrid VAE-GAN Architecture for Artistic Style Transfer`
[paper][].

This paper builds a novel hybrid VAE-GAN method for image to sketch style transfer and modifies the generator, discriminator, encoder and the training procedure to generate good sketches from the corresponding images of the person.

This README file provides brief instructions about how to set up the `Hybrid Architecture` package and reproduce the paper results. 

The code of `Hybrid Architecture` is based on [Hybrid GAN for COVID][].Please refer to the LICENSE section for the proper copyright attribution.

## Applying the Hybrid Architecture to your Dataset

In short, the procedure to adapt the `Hybrid Architecture` to your problem is as follows:

```bash
    PhotoToSketch/      # Name of the dataset
        photos/     # Name of the first domain
        sketches/   # Name of the second domain
```

The root directory is named PhotoToSketch/, representing our dataset.
The two domains are photos/ and sketches/.
Arrange your dataset into a similar form, but choose appropriate names for the dataset directory and data domains.


## Installation and Requirements

To generate sketches from photos using our implementation, follow these steps:

1. Open the FINALIMPLEMENTATION.ipynb file in Google Colab.
2. Make sure you have a Google account and are signed in to access Google Colab.
3. The notebook is set up to run in the Colab environment, which provides necessary GPU resources.

## Running the Implementation

1. Open FINALIMPLEMENTATION.ipynb in Google Colab.
2. Follow the instructions within the notebook to upload your dataset or connect to your Google Drive if your dataset is stored there.
3. Run the cells in order, making adjustments as needed for your specific use case.
4. The notebook will guide you through the process of generating sketches from photos using Keras.

## Additional Resources

1. Alternative Implementations: Check the All-Notebooks folder in our repository to see different versions of notebooks we experimented with during development.
2. CycleGAN Implementation: You can find our CycleGAN implementation in the repository as well, which might be useful for comparison or further experimentation.

## Important Precautions

1. Keras Usage: This implementation uses Keras. Make sure you're familiar with Keras basics or refer to Keras documentation as needed.

2. Colab-Specific Considerations:
    - Colab sessions have time limits. For long-running tasks, you may need to keep the browser tab active or use tricks to prevent disconnection.

    - Colab provides limited GPU time. Monitor your usage to avoid interruptions.

    - Save important outputs or models to your Google Drive, as Colab environments are temporary.

3. Resource Management: Be mindful of resource usage, especially when working with large datasets or models.

4. Data Privacy: If working with sensitive data, be aware of the privacy implications of using cloud-based services like Google Colab.

## Customization

Feel free to modify the `FINALIMPLEMENTATION.ipynb` notebook to suit your specific needs. You can adjust parameters, change the model architecture, or adapt the data processing steps as required.
For any questions or issues, please refer to the documentation within the notebook or raise an issue in the GitHub repository.

## Download Dataset

You can download the dataset using the code provided in the notebook.
You just need to run all the cells.

# FAQs

## General Questions

1. **Q: What does this implementation do?**
   A: This implementation converts photos to sketches using a machine learning model implemented in Keras and run on Google Colab.

2. **Q: Do I need any special hardware to run this?**
   A: No, you don't need special hardware. The implementation runs on Google Colab, which provides the necessary computational resources, including GPUs.

3. **Q: Is this implementation free to use?**
   A: Yes, the implementation itself is free. However, you'll need a Google account to use Colab, which is also free but has usage limits.

## Setup and Requirements

4. **Q: What do I need to get started?**
   A: You need a Google account, access to Google Colab, and your dataset of photos. No local setup is required.

5. **Q: Do I need to install any software on my computer?**
   A: No, everything runs in the cloud on Google Colab. You just need a web browser.

6. **Q: Can I run this on my local machine instead of Colab?**
   A: While it's possible, the notebook is optimized for Colab. Running locally would require setting up the correct environment and potentially modifying the code.

## Usage and Customization

7. **Q: How do I load my own dataset?**
   A: The notebook includes instructions for uploading your dataset directly to Colab or accessing it from Google Drive. Follow these instructions in the notebook.

8. **Q: Can I modify the model or parameters?**
   A: Yes, you can modify the notebook to change model architecture, hyperparameters, or data processing steps. Basic Python and Keras knowledge will be helpful for this.

9. **Q: How long does the process take?**
   A: Processing time depends on your dataset size and the Colab resources allocated. It can range from minutes to hours.

## Troubleshooting

10. **Q: What if I get a "runtime disconnected" error?**
    A: Colab sessions have time limits. For long tasks, keep the browser tab active or use techniques to prevent disconnection, which are discussed in the notebook.

11. **Q: The results don't look good. What can I do?**
    A: Try adjusting the model parameters, preprocessing steps, or training duration. Also, ensure your dataset is diverse and representative.

12. **Q: I'm getting out-of-memory errors. How can I fix this?**
    A: Try reducing batch sizes, using data generators for large datasets, or upgrading to Colab Pro for more resources.

## Data and Privacy

13. **Q: Is my data safe on Google Colab?**
    A: While Colab is generally secure, it's a cloud service. Avoid uploading sensitive or personal data. Refer to Google's privacy policy for more details.

14. **Q: Can I use this for commercial projects?**
    A: Check the license of this implementation and all its dependencies. Also, ensure you have the rights to use your dataset for your intended purpose.

For more specific questions, please refer to the documentation within the notebook or raise an issue in the GitHub repository.

## LICENSE

MIT License.

