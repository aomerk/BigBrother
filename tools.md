# Programmierpraktikum - Sructure and Pointers - SS20


# Notes
All agile, please.

Feel free to reuse across groups.
Please let others reuse.
Please just all contribute. (If you don't, I'll act.)


# Jobs per Project
## RobotDoc
- data
	- for training
		- pre-recorded medical records
		- fed by admins automatically
	- for production
		- pre-recorded medical records (short-term)
			- fed by admins
		- live queried medical records (middle-term)
			- in reality: medical records + ask patient + run some tests on patient
			- for us: medical records + ask patient
- UI
	- code
		- library & application as executable
		- to check out from github
			- git checkout + compile + run the executable on my laptop (short term)
				- it asks for the right rights, and runs actions
				- should run on Mac, Windows and Linux (verify with dockers & CD/CI)
	- program on laptop (long-term)
		- Desktop App
		- package to install from TestPyPI
	- App (long-term)
		- Mobile App
		- on Android/iPhone
	- Web site (long-term)
		- Web App
		- to make work on Chrome					
- design
	- attributes (input)
		- symptoms
		- for inspiration
			- take attributes from existing data
	- classes (output)
		- diseases
		- for inspiration
			- take classes from existing data (output)
	- learner
		- decision tree (short-term)
		- deep neural networks (middle-term)
	- synthetization output text
		- work with RobotDoc and TutorAI
- training
	- get some software pre-trained
	- optional
		- train some more yourself
- User story
	- until confident enough, keep collecting attributes
		- by asking questions in a CLI (short-term)
		- by voice (middle-term), use releases from BetterAlexa

## BetterAlexa
- data
	- for training
		- pre-recorded audio
		- fed by admins automatically
	- for production
		- pre-recorded audio (short-term)
			- fed by admins
		- live audio (middle-term)
			- in reality: to capture from our embedded product
			- for us: to capture from a Web site switching on a camera
- UI
	- code
		- library & application as executable
		- to check out from github
			- git checkout + compile + run the executable on my laptop (short term)
				- it asks for the right rights, and runs actions
				- should run on Mac, Windows and Linux (verify with dockers & CD/CI)
	- program on laptop (middle-term to long-term)
		- Desktop App
		- package to install from TestPyPI
	- App (middle-term to long-term)
		- Mobile App
		- on Android/iPhone
	- Web site (middle-term to long-term)
		- Web App
		- to make work on Chrome
- design
	- attributes (input)
		- take attributes from existing data
	- classes (output)
		- actions (through events)
	- learner
		- existing software
	- synthetization output
		- synthetization output text
			- work with RobotDoc and TutorAI
		- synthetization output voice
			- existing software
- training
	- get some software pre-trained
	- optional
		- train some more yourself
- User story
	1. I tell you I want sth.
		1. Optionally
			- You recognize my voice and authentify me.
	1. You do the action.
		1. You recognize the action.
		1. You do the job.
		1. Optionally
			- You ask me follow-up questions or tell me anything I should know.
	- Actions
		- Playing some music (Spotify)
		- Looking up infos with Google / Wikipedia / Web semantics
			- Where is xxx, who is xxx
		- ISIS
			- opening pages ISIS
		- news
			- crawling for keywords
				- any news on topic XXX
					- meaning I want articles including XXX, no topic analysis here
				- crawl top news sites
		- mail
			- writing up a reply
		- store data
			- cached credentials
				- TUB
				- Google
				- etc.
			- keep safe
				- keep locally
				- encrypt

## BigBrother
- data
	- for training
		- pre-recorded video
		- fed by admins automatically
	- for production
		- pre-recorded video (short-term)
			- fed by admins
		- live stream (middle-term)
			- in reality: to capture from a camera somewhere on the street that connects to the Internet automatically
			- for us: to capture from a Web site switching on a camera
		- multiple live streams (long-term)
			- to aggregate to single analysis central point
- UI
	- code
		- library & application as executable
		- to check out from github
			- git checkout + compile + run the executable on my laptop (short term)
				- it asks for the right rights, and runs actions
				- should run on Mac, Windows and Linux (verify with dockers & CD/CI)	
	- program on laptop (middle-term to long-term)
		- Desktop App
		- package to install from TestPyPI
	- App (middle-term to long-term)
		- Mobile App
		- on Android/iPhone
	- Web site (middle-term to long-term)
		- Web App
		- to make work on Chrome
- design
	- attributes (input)
		- take attributes from existing data
	- classes (output)
		- actions (through events)
	- learner
		- existing software
- training
	- get some software pre-trained
	- optional
		- train some more yourself
- User story
	1. You detect my face.
		- pictures (short-term)
		- videos (middle-term)
			- keep storing data until I stop / I'm away
				- -> better recognition afterwards
		- (Optional: You ask me to fill in my details but otherwise just make me be face XYZ.)
	1. You look me up in your DB
		- per resource queried
			- check resource
				- resource
					- user makes gesture to tell us what resource he wants access to
					- resource =
						- in reality: house/lock
						- for us: just some image
					- example:
						- draw house with hands
							- get access to house
				- single resource (short-term)
				- multiple resources (middle-term)
			- allowing?
				- I'm a bad guy.
					- -> You raise an alarm.
				- I'm a good guy.
					- -> You greet me and give me access to Wonderland.

## TutorAI
- data
	- for training
		- pre-stored message log
		- fed by admins automatically
	- for production
		- pre-recorded message log (short-term)
			- fed by admins
		- live fed message log (middle-term)
			- crawled
- UI
	- code
		- library & application as executable
		- to check out from github
			- git checkout + compile + run the executable on my laptop (short term)
				- it asks for the right rights, and runs actions
				- should run on Mac, Windows and Linux (verify with dockers & CD/CI)
	- program on laptop (long-term)
		- Desktop App
		- package to install from TestPyPI
	- App (long-term)
		- Mobile App
		- on Android/iPhone
	- Web site (long-term)
		- Web App
		- to make work on Chrome					
- design
	- attributes (input)
		- top w.r.t. TF-IDF
		- n-grams
		- for inspiration	
			- take attributes from existing data
	- classes (output)
		- for inspiration
			- take classes from existing data (output)
	- synthetization output text
		- work with RobotDoc and BetterAlexa
	- learner
		- for inspiration	
		- existing software
- training
	- get some software pre-trained
	- optional
		- train some more yourself
- User story
	1. A message is received.
	1. An answer is sent.
	1. Optionally
		- a human is asked for input


# Shortlist of Tools
	- Frontend
		- Text I/O
			- Text Processing: Google API, Sphinx
			- Text Synthetization: Google API, Sphinx
			- Text Acquisition: Google API, Sphinx

		- Audio feature extraction and synthetization
			- Praat, WaveSurfer, audioread
				- Sonic visualizer, Weka

		- Video feature extraction and synthetization
			- Kornia, OpenCV, pytesseract, tesserocr, SimpleCV

		- Text Analysis
			- Syntax Parsing: antlr, stanford-parser

		- Emotion Extraction
			- openaudio, opensmile

		- data visualization
			- Cerberus, colander, jsonschema, schema, Schematics, valideer, voluptuou

	- Backend
		- Stream Processing
			- Apache Flink, Apache Storm

		- ML
			- TensorFlow, Keras, sklearn
				- caffe, keras, mxnet, pytorch, SerpentAI, tensorflow, Theano
				- H2O, Metrics, NuPIC, scikit-learn, Spark ML, vowpal_porpoise, xgboost
		- DBMS
			- Relational
				- MySQL
			- NoSQL
				- MongoDB

	- Glue
		- Web Framework: Django

	- chatbox
		- errbot
			- [useful for TutorAI]

	- news feed management
		- django-activity-stream
		- Stream Framework 



# Long list of Tools

Mostly taken from:
<https://github.com/vinta/awesome-python>

## Asynchronous Programming

* [asyncio](https://docs.python.org/3/library/asyncio.html) - (Python standard library) Asynchronous I/O, event loop, coroutines and tasks.
	- [awesome-asyncio](https://github.com/timofurrer/awesome-asyncio)
* [uvloop](https://github.com/MagicStack/uvloop) - Ultra fast asyncio event loop.
* [Twisted](https://twistedmatrix.com/trac/) - An event-driven networking engine.


## Audio

*Libraries for manipulating audio and its metadata.*

* Audio
	* [audioread](https://github.com/beetbox/audioread) - Cross-library (GStreamer + Core Audio + MAD + FFmpeg) audio decoding.
	* [dejavu](https://github.com/worldveil/dejavu) - Audio fingerprinting and recognition.
	* [matchering](https://github.com/sergree/matchering) - A library for automated reference audio mastering.
	* [mingus](http://bspaans.github.io/python-mingus/) - An advanced music theory and notation package with MIDI file and playback support.
	* [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) - Audio feature extraction, classification, segmentation and applications.
	* [pydub](https://github.com/jiaaro/pydub) - Manipulate audio with a simple and easy high level interface.
	* [TimeSide](https://github.com/Parisson/TimeSide) - Open web audio processing framework.
* Metadata
	* [beets](https://github.com/beetbox/beets) - A music library manager and [MusicBrainz](https://musicbrainz.org/) tagger.
	* [eyeD3](https://github.com/nicfit/eyeD3) - A tool for working with audio files, specifically MP3 files containing ID3 metadata.
	* [mutagen](https://github.com/quodlibet/mutagen) - A Python module to handle audio metadata.
	* [tinytag](https://github.com/devsnd/tinytag) - A library for reading music meta data of MP3, OGG, FLAC and Wave files.


## CMS

*Content Management Systems.*

* [wagtail](https://wagtail.io/) - A Django content management system.
* [django-cms](https://www.django-cms.org/en/) - An Open source enterprise CMS based on the Django.
* [feincms](https://github.com/feincms/feincms) - One of the most advanced Content Management Systems built on Django.
* [indico](https://github.com/indico/indico) - A feature-rich event management system, made @ [CERN](https://en.wikipedia.org/wiki/CERN).
* [Kotti](https://github.com/Kotti/Kotti) - A high-level, Pythonic web application framework built on Pyramid.
* [mezzanine](https://github.com/stephenmcd/mezzanine) - A powerful, consistent, and flexible content management platform.
* [plone](https://plone.org/) - A CMS built on top of the open source application server Zope.
* [quokka](https://github.com/rochacbruno/quokka) - Flexible, extensible, small CMS powered by Flask and MongoDB.


## Computer Vision

*Libraries for Computer Vision.*

* [Kornia](https://github.com/kornia/kornia/) - Open Source Differentiable Computer Vision Library for PyTorch.
* [OpenCV](https://opencv.org/) - Open Source Computer Vision Library.
* [pytesseract](https://github.com/madmaze/pytesseract) - Another wrapper for [Google Tesseract OCR](https://github.com/tesseract-ocr).
* [tesserocr](https://github.com/sirfz/tesserocr) - A simple, Pillow-friendly, wrapper around the `tesseract-ocr` API for OCR.
* [SimpleCV](https://github.com/sightmachine/SimpleCV) - An open source framework for building computer vision applications.


## Data Analysis

*Libraries for data analyzing.*

* [Blaze](https://github.com/blaze/blaze) - NumPy and Pandas interface to Big Data.
* [Open Mining](https://github.com/mining/mining) - Business Intelligence (BI) in Pandas interface.
* [Orange](https://orange.biolab.si/) - Data mining, data visualization, analysis and machine learning through visual programming or scripts.
* [Pandas](http://pandas.pydata.org/) - A library providing high-performance, easy-to-use data structures and data analysis tools.
* [Optimus](https://github.com/ironmussa/Optimus) - Agile Data Science Workflows made easy with PySpark.



## Data Visualization

*Libraries for visualizing data. Also see [awesome-javascript](https://github.com/sorrycc/awesome-javascript#data-visualization).*

* [Altair](https://github.com/altair-viz/altair) - Declarative statistical visualization library for Python.
* [Bokeh](https://github.com/bokeh/bokeh) - Interactive Web Plotting for Python.
* [bqplot](https://github.com/bloomberg/bqplot) - Interactive Plotting Library for the Jupyter Notebook
* [Dash](https://plot.ly/products/dash/) - Built on top of Flask, React and Plotly aimed at analytical web applications.
	* [awesome-dash](https://github.com/Acrotrend/awesome-dash)
* [diagrams](https://github.com/mingrammer/diagrams) - Diagram as Code.
* [plotnine](https://github.com/has2k1/plotnine) - A grammar of graphics for Python based on ggplot2.
* [Matplotlib](http://matplotlib.org/) - A Python 2D plotting library.
* [Pygal](http://www.pygal.org/en/latest/) - A Python SVG Charts Creator.
* [PyGraphviz](https://pypi.org/project/pygraphviz/) - Python interface to [Graphviz](http://www.graphviz.org/).
* [PyQtGraph](http://www.pyqtgraph.org/) - Interactive and realtime 2D/3D/Image plotting and science/engineering widgets.
* [Seaborn](https://github.com/mwaskom/seaborn) - Statistical data visualization using Matplotlib.
* [VisPy](https://github.com/vispy/vispy) - High-performance scientific visualization based on OpenGL.


## Database

*Databases implemented in Python.*

* [pickleDB](https://github.com/patx/pickledb) - A simple and lightweight key-value store for Python.
* [tinydb](https://github.com/msiemens/tinydb) - A tiny, document-oriented database.
* [ZODB](https://github.com/zopefoundation/ZODB) - A native object database for Python. A key-value and object graph database.


## Database Drivers

*Libraries for connecting and operating databases.*

* MySQL - [awesome-mysql](http://shlomi-noach.github.io/awesome-mysql/)
	* [mysqlclient](https://github.com/PyMySQL/mysqlclient-python) - MySQL connector with Python 3 support ([mysql-python](https://sourceforge.net/projects/mysql-python/) fork).
	* [PyMySQL](https://github.com/PyMySQL/PyMySQL) - A pure Python MySQL driver compatible to mysql-python.
* PostgreSQL - [awesome-postgres](https://github.com/dhamaniasad/awesome-postgres)
	* [psycopg2](http://initd.org/psycopg/) - The most popular PostgreSQL adapter for Python.
	* [queries](https://github.com/gmr/queries) - A wrapper of the psycopg2 library for interacting with PostgreSQL.
* Other Relational Databases
	* [pymssql](http://www.pymssql.org/en/latest/) - A simple database interface to Microsoft SQL Server.
	* [SuperSQLite](https://github.com/plasticityai/supersqlite) - A supercharged SQLite library built on top of [apsw](https://github.com/rogerbinns/apsw).
* NoSQL Databases
	* [cassandra-driver](https://github.com/datastax/python-driver) - The Python Driver for Apache Cassandra.
	* [happybase](https://github.com/wbolster/happybase) - A developer-friendly library for Apache HBase.
	* [kafka-python](https://github.com/dpkp/kafka-python) - The Python client for Apache Kafka.
	* [py2neo](https://py2neo.org/) - A client library and toolkit for working with Neo4j.
	* [pymongo](https://github.com/mongodb/mongo-python-driver) - The official Python client for MongoDB.
	* [redis-py](https://github.com/andymccurdy/redis-py) - The Python client for Redis.
* Asynchronous Clients
	* [motor](https://github.com/mongodb/motor) - The async Python driver for MongoDB.


## Deep Learning

*Frameworks for Neural Networks and Deep Learning. Also see [awesome-deep-learning](https://github.com/ChristosChristofidis/awesome-deep-learning).*

* [caffe](https://github.com/BVLC/caffe) - A fast open framework for deep learning..
* [keras](https://github.com/keras-team/keras) - A high-level neural networks library and capable of running on top of either TensorFlow or Theano.
* [mxnet](https://github.com/dmlc/mxnet) - A deep learning framework designed for both efficiency and flexibility.
* [pytorch](https://github.com/pytorch/pytorch) - Tensors and Dynamic neural networks in Python with strong GPU acceleration.
* [SerpentAI](https://github.com/SerpentAI/SerpentAI) - Game agent framework. Use any video game as a deep learning sandbox.
* [tensorflow](https://github.com/tensorflow/tensorflow) - The most popular Deep Learning framework created by Google.
* [Theano](https://github.com/Theano/Theano) - A library for fast numerical computation.


## GUI Development

*Libraries for working with graphical user interface applications.*

* [curses](https://docs.python.org/3/library/curses.html) - Built-in wrapper for [ncurses](http://www.gnu.org/software/ncurses/) used to create terminal GUI applications.
* [Eel](https://github.com/ChrisKnott/Eel) - A library for making simple Electron-like offline HTML/JS GUI apps.
* [enaml](https://github.com/nucleic/enaml) - Creating beautiful user-interfaces with Declarative Syntax like QML.
* [Flexx](https://github.com/zoofIO/flexx) - Flexx is a pure Python toolkit for creating GUI's, that uses web technology for its rendering.
* [Gooey](https://github.com/chriskiehl/Gooey) - Turn command line programs into a full GUI application with one line.
* [kivy](https://kivy.org/) - A library for creating NUI applications, running on Windows, Linux, Mac OS X, Android and iOS.
* [pyglet](https://bitbucket.org/pyglet/pyglet/wiki/Home) - A cross-platform windowing and multimedia library for Python.
* [PyGObject](https://wiki.gnome.org/Projects/PyGObject) - Python Bindings for GLib/GObject/GIO/GTK+ (GTK+3).
* [PyQt](https://riverbankcomputing.com/software/pyqt/intro) - Python bindings for the [Qt](https://www.qt.io/) cross-platform application and UI framework.
* [PySimpleGUI](https://github.com/PySimpleGUI/PySimpleGUI) - Wrapper for tkinter, Qt, WxPython and Remi.
* [pywebview](https://github.com/r0x0r/pywebview/) - A lightweight cross-platform native wrapper around a webview component.
* [Tkinter](https://wiki.python.org/moin/TkInter) - Tkinter is Python's de-facto standard GUI package.
* [Toga](https://github.com/pybee/toga) - A Python native, OS native GUI toolkit.
* [urwid](http://urwid.org/) - A library for creating terminal GUI applications with strong support for widgets, events, rich colors, etc.
* [wxPython](https://wxpython.org/) - A blending of the wxWidgets C++ class library with the Python.


## HTTP Clients

*Libraries for working with HTTP.*

* [grequests](https://github.com/spyoungtech/grequests) - requests + gevent for asynchronous HTTP requests.
* [httplib2](https://github.com/httplib2/httplib2) - Comprehensive HTTP client library.
* [httpx](https://github.com/encode/httpx) - A next generation HTTP client for Python.
* [requests](https://github.com/psf/requests) - HTTP Requests for Humans.
* [treq](https://github.com/twisted/treq) - Python requests like API built on top of Twisted's HTTP client.
* [urllib3](https://github.com/shazow/urllib3) - A HTTP library with thread-safe connection pooling, file post support, sanity friendly.


## Image Processing

*Libraries for manipulating images.*

* [hmap](https://github.com/rossgoodwin/hmap) - Image histogram remapping.
* [imgSeek](https://sourceforge.net/projects/imgseek/) - A project for searching a collection of images using visual similarity.
* [nude.py](https://github.com/hhatto/nude.py) - Nudity detection.
* [pagan](https://github.com/daboth/pagan) - Retro identicon (Avatar) generation based on input string and hash.
* [pillow](https://github.com/python-pillow/Pillow) - Pillow is the friendly [PIL](http://www.pythonware.com/products/pil/) fork.
* [pyBarcode](https://pythonhosted.org/pyBarcode/) - Create barcodes in Python without needing PIL.
* [pygram](https://github.com/ajkumar25/pygram) - Instagram-like image filters.
* [python-qrcode](https://github.com/lincolnloop/python-qrcode) - A pure Python QR Code generator.
* [Quads](https://github.com/fogleman/Quads) - Computer art based on quadtrees.
* [scikit-image](http://scikit-image.org/) - A Python library for (scientific) image processing.
* [thumbor](https://github.com/thumbor/thumbor) - A smart imaging service. It enables on-demand crop, re-sizing and flipping of images.
* [wand](https://github.com/dahlia/wand) - Python bindings for [MagickWand](http://www.imagemagick.org/script/magick-wand.php), C API for ImageMagick.


## Machine Learning

*Libraries for Machine Learning. Also see [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning#python).*

* [H2O](https://github.com/h2oai/h2o-3) - Open Source Fast Scalable Machine Learning Platform.
* [Metrics](https://github.com/benhamner/Metrics) - Machine learning evaluation metrics.
* [NuPIC](https://github.com/numenta/nupic) - Numenta Platform for Intelligent Computing.
* [scikit-learn](http://scikit-learn.org/) - The most popular Python library for Machine Learning.
* [Spark ML](http://spark.apache.org/docs/latest/ml-guide.html) - [Apache Spark](http://spark.apache.org/)'s scalable Machine Learning library.
* [vowpal_porpoise](https://github.com/josephreisinger/vowpal_porpoise) - A lightweight Python wrapper for [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit/).
* [xgboost](https://github.com/dmlc/xgboost) - A scalable, portable, and distributed gradient boosting library.


## Natural Language Processing

*Libraries for working with human languages.*

- General
	* [gensim](https://github.com/RaRe-Technologies/gensim) - Topic Modeling for Humans.
	* [langid.py](https://github.com/saffsd/langid.py) - Stand-alone language identification system.
	* [nltk](http://www.nltk.org/) - A leading platform for building Python programs to work with human language data.
	* [pattern](https://github.com/clips/pattern) - A web mining module.
	* [polyglot](https://github.com/aboSamoor/polyglot) - Natural language pipeline supporting hundreds of languages.
	* [pytext](https://github.com/facebookresearch/pytext) - A natural language modeling framework based on PyTorch.
	* [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP) - A toolkit enabling rapid deep learning NLP prototyping for research.
	* [spacy](https://spacy.io/) - A library for industrial-strength natural language processing in Python and Cython.
	* [Stanza](https://github.com/stanfordnlp/stanza) - The Stanford NLP Group's official Python library, supporting 60+ languages.
- Chinese
	* [jieba](https://github.com/fxsjy/jieba) - The most popular Chinese text segmentation library.
	* [pkuseg-python](https://github.com/lancopku/pkuseg-python) - A toolkit for Chinese word segmentation in various domains.
	* [snownlp](https://github.com/isnowfy/snownlp) - A library for processing Chinese text.
	* [funNLP](https://github.com/fighting41love/funNLP) - A collection of tools and datasets for Chinese NLP.


## News Feed

*Libraries for building user's activities.*

* [django-activity-stream](https://github.com/justquick/django-activity-stream) - Generating generic activity streams from the actions on your site.
* [Stream Framework](https://github.com/tschellenbach/Stream-Framework) - Building news feed and notification systems using Cassandra and Redis.


## ORM

*Libraries that implement Object-Relational Mapping or data mapping techniques.*

* Relational Databases
	* [Django Models](https://docs.djangoproject.com/en/dev/topics/db/models/) - The Django ORM.
	* [SQLAlchemy](https://www.sqlalchemy.org/) - The Python SQL Toolkit and Object Relational Mapper.
		* [awesome-sqlalchemy](https://github.com/dahlia/awesome-sqlalchemy)
	* [dataset](https://github.com/pudo/dataset) - Store Python dicts in a database - works with SQLite, MySQL, and PostgreSQL.
	* [orator](https://github.com/sdispater/orator) -  The Orator ORM provides a simple yet beautiful ActiveRecord implementation.
	* [orm](https://github.com/encode/orm) - An async ORM.
	* [peewee](https://github.com/coleifer/peewee) - A small, expressive ORM.
	* [pony](https://github.com/ponyorm/pony/) - ORM that provides a generator-oriented interface to SQL.
	* [pydal](https://github.com/web2py/pydal/) - A pure Python Database Abstraction Layer.
* NoSQL Databases
	* [hot-redis](https://github.com/stephenmcd/hot-redis) - Rich Python data types for Redis.
	* [mongoengine](https://github.com/MongoEngine/mongoengine) - A Python Object-Document-Mapper for working with MongoDB.
	* [PynamoDB](https://github.com/pynamodb/PynamoDB) - A Pythonic interface for [Amazon DynamoDB](https://aws.amazon.com/dynamodb/).
	* [redisco](https://github.com/kiddouk/redisco) - A Python Library for Simple Models and Containers Persisted in Redis.


## Recommender Systems

*Libraries for building recommender systems.*

* [annoy](https://github.com/spotify/annoy) - Approximate Nearest Neighbors in C++/Python optimized for memory usage.
* [fastFM](https://github.com/ibayer/fastFM) - A library for Factorization Machines.
* [implicit](https://github.com/benfred/implicit) - A fast Python implementation of collaborative filtering for implicit datasets.
* [libffm](https://github.com/guestwalk/libffm) - A library for Field-aware Factorization Machine (FFM).
* [lightfm](https://github.com/lyst/lightfm) - A Python implementation of a number of popular recommendation algorithms.
* [spotlight](https://github.com/maciejkula/spotlight) - Deep recommender models using PyTorch.
* [Surprise](https://github.com/NicolasHug/Surprise) - A scikit for building and analyzing recommender systems.
* [tensorrec](https://github.com/jfkirk/tensorrec) - A Recommendation Engine Framework in TensorFlow.


## RESTful API

*Libraries for building RESTful APIs.*

* Django
	* [django-rest-framework](http://www.django-rest-framework.org/) - A powerful and flexible toolkit to build web APIs.
	* [django-tastypie](http://tastypieapi.org/) - Creating delicious APIs for Django apps.
* Flask
	* [eve](https://github.com/pyeve/eve) - REST API framework powered by Flask, MongoDB and good intentions.
	* [flask-api](https://github.com/flask-api/flask-api) - Browsable Web APIs for Flask.
	* [flask-restful](https://github.com/flask-restful/flask-restful) - Quickly building REST APIs for Flask.
* Pyramid
	* [cornice](https://github.com/Cornices/cornice) - A RESTful framework for Pyramid.
* Framework agnostic
	* [apistar](https://github.com/encode/apistar) - A smart Web API framework, designed for Python 3.
	* [falcon](https://github.com/falconry/falcon) - A high-performance framework for building cloud APIs and web app backends.
	* [fastapi](https://github.com/tiangolo/fastapi) - A modern, fast, web framework for building APIs with Python 3.6+ based on standard Python type hints.
	* [hug](https://github.com/hugapi/hug) - A Python 3 framework for cleanly exposing APIs.
	* [sandman2](https://github.com/jeffknupp/sandman2) - Automated REST APIs for existing database-driven systems.
	* [sanic](https://github.com/huge-success/sanic) - A Python 3.6+ web server and web framework that's written to go fast.
	* [vibora](https://vibora.io/) - Fast, efficient and asynchronous Web framework inspired by Flask.


## RPC Servers

*RPC-compatible servers.*

* [zeroRPC](https://github.com/0rpc/zerorpc-python) - zerorpc is a flexible RPC implementation based on [ZeroMQ](http://zeromq.org/) and [MessagePack](http://msgpack.org/).
* [RPyC](https://github.com/tomerfiliba/rpyc) (Remote Python Call) - A transparent and symmetric RPC library for Python


## Search

*Libraries and software for indexing and performing search queries on data.*

* [elasticsearch-py](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/index.html) - The official low-level Python client for [Elasticsearch](https://www.elastic.co/products/elasticsearch).
* [elasticsearch-dsl-py](https://github.com/elastic/elasticsearch-dsl-py) - The official high-level Python client for Elasticsearch.
* [django-haystack](https://github.com/django-haystack/django-haystack) - Modular search for Django.
* [pysolr](https://github.com/django-haystack/pysolr) - A lightweight Python wrapper for [Apache Solr](https://lucene.apache.org/solr/).
* [whoosh](http://whoosh.readthedocs.io/en/latest/) - A fast, pure Python search engine library.


## Serverless Frameworks

*Frameworks for developing serverless Python code.*

* [python-lambda](https://github.com/nficano/python-lambda) - A toolkit for developing and deploying Python code in AWS Lambda.
* [Zappa](https://github.com/Miserlou/Zappa) - A tool for deploying WSGI applications on AWS Lambda and API Gateway.


## Task Queues

*Libraries for working with task queues.*

* [celery](http://www.celeryproject.org/) - An asynchronous task queue/job queue based on distributed message passing.
* [huey](https://github.com/coleifer/huey) - Little multi-threaded task queue.
* [mrq](https://github.com/pricingassistant/mrq) - A distributed worker task queue in Python using Redis & gevent.
* [rq](https://github.com/rq/rq) - Simple job queues for Python.


## Text Processing

*Libraries for parsing and manipulating plain texts.*

* General
	* [chardet](https://github.com/chardet/chardet) - Python 2/3 compatible character encoding detector.
	* [difflib](https://docs.python.org/3/library/difflib.html) - (Python standard library) Helpers for computing deltas.
	* [ftfy](https://github.com/LuminosoInsight/python-ftfy) - Makes Unicode text less broken and more consistent automagically.
	* [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy) - Fuzzy String Matching.
	* [Levenshtein](https://github.com/ztane/python-Levenshtein/) - Fast computation of Levenshtein distance and string similarity.
	* [pangu.py](https://github.com/vinta/pangu.py) - Paranoid text spacing.
	* [pyfiglet](https://github.com/pwaller/pyfiglet) - An implementation of figlet written in Python.
	* [pypinyin](https://github.com/mozillazg/python-pinyin) - Convert Chinese hanzi (漢字) to pinyin (拼音).
	* [textdistance](https://github.com/orsinium/textdistance) - Compute distance between sequences with 30+ algorithms.
	* [unidecode](https://pypi.org/project/Unidecode/) - ASCII transliterations of Unicode text.
* Slugify
	* [awesome-slugify](https://github.com/dimka665/awesome-slugify) - A Python slugify library that can preserve unicode.
	* [python-slugify](https://github.com/un33k/python-slugify) - A Python slugify library that translates unicode to ASCII.
	* [unicode-slugify](https://github.com/mozilla/unicode-slugify) - A slugifier that generates unicode slugs with Django as a dependency.
* Unique identifiers
	* [hashids](https://github.com/davidaurelio/hashids-python) - Implementation of [hashids](http://hashids.org) in Python.
	* [shortuuid](https://github.com/skorokithakis/shortuuid) - A generator library for concise, unambiguous and URL-safe UUIDs.
* Parser
	* [ply](https://github.com/dabeaz/ply) - Implementation of lex and yacc parsing tools for Python.
	* [pygments](http://pygments.org/) - A generic syntax highlighter.
	* [pyparsing](https://github.com/pyparsing/pyparsing) - A general purpose framework for generating parsers.
	* [python-nameparser](https://github.com/derek73/python-nameparser) - Parsing human names into their individual components.
	* [python-phonenumbers](https://github.com/daviddrysdale/python-phonenumbers) - Parsing, formatting, storing and validating international phone numbers.
	* [python-user-agents](https://github.com/selwin/python-user-agents) - Browser user agent parser.
	* [sqlparse](https://github.com/andialbrecht/sqlparse) - A non-validating SQL parser.


## Third-party APIs

*Libraries for accessing third party services APIs. Also see [List of Python API Wrappers and Libraries](https://github.com/realpython/list-of-python-api-wrappers).*

* [apache-libcloud](https://libcloud.apache.org/) - One Python library for all clouds.
* [boto3](https://github.com/boto/boto3) - Python interface to Amazon Web Services.
* [django-wordpress](https://github.com/istrategylabs/django-wordpress) - WordPress models and views for Django.
* [facebook-sdk](https://github.com/mobolic/facebook-sdk) - Facebook Platform Python SDK.
* [google-api-python-client](https://github.com/google/google-api-python-client) - Google APIs Client Library for Python.
* [gspread](https://github.com/burnash/gspread) - Google Spreadsheets Python API.
* [twython](https://github.com/ryanmcgrath/twython) - A Python wrapper for the Twitter API.


## URL Manipulation

*Libraries for parsing URLs.*

* [furl](https://github.com/gruns/furl) - A small Python library that makes parsing and manipulating URLs easy.
* [purl](https://github.com/codeinthehole/purl) - A simple, immutable URL class with a clean API for interrogation and manipulation.
* [pyshorteners](https://github.com/ellisonleao/pyshorteners) - A pure Python URL shortening lib.
* [webargs](https://github.com/marshmallow-code/webargs) - A friendly library for parsing HTTP request arguments with built-in support for popular web frameworks.


## Video

*Libraries for manipulating video and GIFs.*

* [vidgear](https://github.com/abhiTronix/vidgear) - Most Powerful multi-threaded Video Processing framework.
* [moviepy](https://zulko.github.io/moviepy/) - A module for script-based movie editing with many formats, including animated GIFs.
* [scikit-video](https://github.com/aizvorski/scikit-video) - Video processing routines for SciPy.


## Web Content Extracting

*Libraries for extracting web contents.*

* [html2text](https://github.com/Alir3z4/html2text) - Convert HTML to Markdown-formatted text.
* [lassie](https://github.com/michaelhelmick/lassie) - Web Content Retrieval for Humans.
* [micawber](https://github.com/coleifer/micawber) - A small library for extracting rich content from URLs.
* [newspaper](https://github.com/codelucas/newspaper) - News extraction, article extraction and content curation in Python.
* [python-readability](https://github.com/buriy/python-readability) - Fast Python port of arc90's readability tool.
* [requests-html](https://github.com/psf/requests-html) - Pythonic HTML Parsing for Humans.
* [sumy](https://github.com/miso-belica/sumy) - A module for automatic summarization of text documents and HTML pages.
* [textract](https://github.com/deanmalmgren/textract) - Extract text from any document, Word, PowerPoint, PDFs, etc.
* [toapi](https://github.com/gaojiuli/toapi) - Every web site provides APIs.


## Web Crawling

*Libraries to automate web scraping.*

* [cola](https://github.com/chineking/cola) - A distributed crawling framework.
* [feedparser](https://pythonhosted.org/feedparser/) - Universal feed parser.
* [grab](https://github.com/lorien/grab) - Site scraping framework.
* [MechanicalSoup](https://github.com/MechanicalSoup/MechanicalSoup) - A Python library for automating interaction with websites.
* [pyspider](https://github.com/binux/pyspider) - A powerful spider system.
* [robobrowser](https://github.com/jmcarp/robobrowser) - A simple, Pythonic library for browsing the web without a standalone web browser.
* [scrapy](https://scrapy.org/) - A fast high-level screen scraping and web crawling framework.
* [portia](https://github.com/scrapinghub/portia) - Visual scraping for Scrapy.


## Web Frameworks

*Traditional full stack web frameworks. Also see [RESTful API](https://github.com/vinta/awesome-python#restful-api)*

* Synchronous
	* [Django](https://www.djangoproject.com/) - The most popular web framework in Python.
		* [awesome-django](https://github.com/shahraizali/awesome-django)
	* [Flask](http://flask.pocoo.org/) - A microframework for Python.
		* [awesome-flask](https://github.com/humiaozuzu/awesome-flask)
	* [Pyramid](https://pylonsproject.org/) - A small, fast, down-to-earth, open source Python web framework.
		* [awesome-pyramid](https://github.com/uralbash/awesome-pyramid)
	* [Masonite](https://github.com/MasoniteFramework/masonite) - The modern and developer centric Python web framework.
* Asynchronous
	* [Tornado](http://www.tornadoweb.org/en/latest/) - A web framework and asynchronous networking library.


## WebSocket

*Libraries for working with WebSocket.*

* [autobahn-python](https://github.com/crossbario/autobahn-python) - WebSocket & WAMP for Python on Twisted and [asyncio](https://docs.python.org/3/library/asyncio.html).
* [channels](https://github.com/django/channels) - Developer-friendly asynchrony for Django.
* [websockets](https://github.com/aaugustin/websockets) - A library for building WebSocket servers and clients with a focus on correctness and simplicity.


## WSGI Servers

*WSGI-compatible web servers.*

* [bjoern](https://github.com/jonashaag/bjoern) - Asynchronous, very fast and written in C.
* [gunicorn](https://github.com/benoitc/gunicorn) - Pre-forked, partly written in C.
* [uWSGI](https://uwsgi-docs.readthedocs.io/en/latest/) - A project aims at developing a full stack for building hosting services, written in C.
* [waitress](https://github.com/Pylons/waitress) - Multi-threaded, powers Pyramid.
* [werkzeug](https://github.com/pallets/werkzeug) - A WSGI utility library for Python that powers Flask and can easily be embedded into your own projects.

### Links
- <https://github.com/EthicalML/awesome-production-machine-learning>
