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

In order to index authors conviently, an optional argument can be added:
* id: id of authors.

The API needs to be luanched before accessed.

### Launch API

Luanch API with the following command line after navigating to **'src'** folder:

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

Access API on **'HOST:PORT/judge'** using HTTP **POST** Method, with all 5 statistics (features) as payload in JSON format. (*id* can be added.)

Predicted label, probabilities of prediction *(and id if available)* are returned in JSON format.

'label' = 1 if author is predicted as student, otherwise label = 0 if predicted as non-student.

Examples 1: (Requests >= 2.19.1)

```
import requests

url = 'http://127.0.0.1:5000/judge'
payload = [{'pc': 303, 'cn': 11111, 'hi': 55, 'gi': 99, 'year_range': 20}]

response = requests.post(url, json = payload)

print(response.text)
```

Results:

```
[
    {
        "label": 0,
        "probability": 0.999997735
    }
]
```

Examples 2:

```
import requests

url = 'http://127.0.0.1:5000/judge'
payload = [{'id': 0, 'pc': 16, 'cn': 1788, 'hi': 13, 'gi': 16, 'year_range': 9}, 
           {'id': 1, 'pc': 9, 'cn': 579, 'hi': 5, 'gi': 9, 'year_range': 8}]

response = requests.post(url, json = payload)

print(response.text)
```

Results:

```
[
    {
        "id": 0,
        "label": 1,
        "probability": 0.5169955492
    },
    {
        "id": 1,
        "label": 1,
        "probability": 0.9957641363
    }
]
```

## Authors

* **Junlin Song** - *Initial work* - [julius-song](https://github.com/julius-song)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

