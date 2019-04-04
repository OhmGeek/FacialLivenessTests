var express = require('express');
var bodyParser = require('body-parser');
const axios = require('axios');

var path = require('path');
var app = express();

var server = require('http').Server(app);

// TODO: insert the URLs to poll here, including the route.
const QualityLivenessURL = ""
const CNN2DLivenessURL = ""

app.use(bodyParser.json()); // support json encoded bodies
app.use(bodyParser.urlencoded({
    extended: true
})); // support encoded body.

var currentDirectory = (process.env.PORT) ? process.cwd() : __dirname;
app.set("port", process.env.PORT || 3000);

function QualityLivenessTest(img) {
    return axios.post(QualityLivenessURL, {img: img});
}

function CNN2DLivenessTest(img) {
    return axios.post(CNN2DLivenessURL, {img: img});
}

app.post('/getLiveness', (req, res) => {
    let img = req.body(); // The body is simply the image (base64 encoded). That's all.
    
    let qualityTest = QualityLivenessTest(img);
    let cnn2DLivenessTest = CNN2DLivenessTest(img);

    // Combine them together.
    let combined = Promise.all([qualityTest, cnn2DLivenessTest]);

    // Once they are both satisfied, return the combined result.
    combined.then((output) => {
        res.send(output);
    }).catch((err) => {
        console.error(err);
    })

})

// ----------- STATIC CONTENT --------------
app.use(express.static(path.join(currentDirectory, "client")));
app.use(express.static(path.join(currentDirectory, "client")));
app.get("*", function(req, res) {
    res.status(404).send("File not found");
});

server.listen(app.get("port"), function() {
    console.log("Server started on port " + app.get("port"));
});
