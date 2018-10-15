# Student Identity Judgement API

An API for judging student identity of authors.

## Requirements

The following environment or packages need to be properly installed.

```
Python >= 3.6
Flask >= 1.0.2
Tensorflow >= 1.10.0
Pandas >= 0.23.4
Numpy >= 1.15.2
```

## Usage

To judge the student identity of a certain author, 5 statistics are needed:
* pc: total number of publications;
* cn: total number of citations;
* hi: H-index;
* gi: G-index;
* year_range: time range from the first to the last publication. 
    *  *year_range = year of latest publication - year of earlist publication + 1*

the API needs to be luanched before accessed.

### Launch API

Luanch API with the following command line after navigating to 'src' folder:

```
$ python api.py [--classifier CLASSIFIER] [--host HOST] [--port PORT]
```

With arguments:

```
--classifier    Choose which classifier to use, dnn_classifier or linear_classifier, default dnn_classifier.
--host          Host url of API, default localhost.
--port          Port of the host API used, defulat 5000.
```

### Access API

Access API using GET Method.

```
Give an example
```

## Authors

* **Junlin Song** - *Initial work* - [julius-song](https://github.com/julius-song)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

