# Recommender System Generalize Framework

## Overview
The Recommender System Generic Framework is a flexible and extensible platform for building recommendation systems. It provides a foundation upon which various recommendation algorithms can be implemented and tested. This README serves as a guide to understanding the framework, its components, and how to use it effectively.

## Features
- Modular architecture allowing easy integration of different recommendation algorithms.
- Supports both user-based and item-based recommendation techniques.
- Configurable parameters for fine-tuning recommendation algorithms.
- Extensive documentation and example code for quick start and customization.
- Compatibility with various data formats, including CSV, JSON, and database connections.

## Getting Started
Follow these steps to get started with the Recommender System Generic Framework:

1. **Installation**: Clone the repository to your local machine.
  ```
  git clone https://github.com/deepakdhull80/reco-framework.git
  ```

3. **Dependencies**: Install the required dependencies using pip.
  ```
  pip install -r requirements.txt
  ```


5. **Usage**: Explore the example scripts provided in the `examples` directory to understand how to use the framework. Customize the scripts according to your specific requirements.

6. **Documentation**: Refer to the documentation in the `docs` directory for detailed information about the framework's architecture, APIs, and usage guidelines.

## Directory Structure(TODO)
- `docs/`: Documentation directory containing detailed information about the framework and its components.

## TODO's:
- [ ] Setup SimpleTrainingPipeline *
- [ ] Create Stats class: compute feature stats, like min, max, mean of numerical features.
- [ ] Read config from yaml file and create its parser class.
- [ ] Modularize data reading and processing.
- [ ] Add dataloader class
- [ ] Add training Strategies: local system, remote
    Accelerate, all_reduce for dpp multi gpu. Single gpu strategy.
- [ ] Add trainer class
- [ ] Tracking and logging with wandb, tensorboard.
- [ ] Add evaluator for ranking and retrieval objectives.
- [ ] Add pytorch lightening support.

## Completed:
- [x] Setup hydra config
    Structure based upon the components: (like train, eval, model, dataloader, inference)
- [x] Create abstract classes
- [x] Create feature config class
- [x] Create model config class
## Contribution Guidelines
We welcome contributions from the community to enhance the features and usability of the Recommender System Generic Framework. If you'd like to contribute, please follow these guidelines:

- Fork the repository and create a new branch for your contribution.
- Make your changes and submit a pull request, providing a detailed description of the changes made.
- Ensure that your code adheres to the coding standards and is well-documented.
- Test your changes thoroughly before submitting the pull request.

## License
The Recommender System Generic Framework is open-source software licensed under the [MIT License](LICENSE).

## Support
For any questions, issues, or feature requests, please [open an issue](https://github.com/deepakdhull80/reco-framework/issues) on GitHub. We'll be happy to assist you.

## Credits
The Recommender System Generic Framework is developed and maintained by [Deepak Dhull]. We would like to thank all contributors for their valuable contributions to the project.


Reference:
- [DLRM - Meta](https://github.com/facebookresearch/dlrm )