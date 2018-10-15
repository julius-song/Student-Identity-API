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

To judge the student identity of a certain author, 5 statistics (features) are needed:
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

Access API on 'HOST:PORT/judge' using HTTP GET Method, with all 5 statistics (features) as query parameters.

Author features and predicted label are returned in JSON format.

'label' = 1 if author is predicted as student, otherwise label = 0 if predicted as non-student.

Examples:

```
import requests

url = 'http://127.0.0.1:5000/judge'
parameters = {'pc': 303, 'cn': 11111, 'hi': 55, 'gi': 99, 'year_range': 20}

response = requests.get(url, params = parameters)

print(response.text)
```

Results:

```
{
  "features": {
    "cn": 11111, 
    "gi": 99, 
    "hi": 55, 
    "pc": 303, 
    "year_range": 20
  }, 
  "label": 0
}
```

## Authors

* **Junlin Song** - *Initial work* - [julius-song](https://github.com/julius-song)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

