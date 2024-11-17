// Attacks on Secure Implementations Assignment 1
// For more info: https://iss.oy.ne.ro/Attacks

'use strict';
var express = require('express');
var app = express();
var sha1=require('sha1');
var bigInt = require("big-integer");

const DIFFICULT_PASSWORD_SALT = "no_secrets";
const BASE_TIME_TO_STALL = 50;

// Source: https://www.npmjs.com/package/sleep
var sleepArray = new Int32Array(new SharedArrayBuffer(4));
function msleep(n) {
    if (n>0) {
        Atomics.wait(sleepArray, 0, 0, n);
    }
}

function sleep(n) {
  msleep(n*1000);
}

function getDifficultPasswordForUser(username,difficulty) {
  // Hash the username
  var hashedUserName = sha1(DIFFICULT_PASSWORD_SALT + difficulty + username);

  // Convert to an a-z string
  var hashedUserNameAsString = bigInt(hashedUserName,16).toString(26,"abcdefghijklmnopqrstuvwxyz");

  return hashedUserNameAsString.substring(0,16);
}

// Gaussian variable with mean 0 and variance 1
// Source: https://stackoverflow.com/a/36481059
function randn_bm() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}


var secretListOfPasswords = ['parshandata', 'dalphon', 'aspata', 'porata', 'adalya', 'aridata',
'parmashta', 'arisai', 'aridai', 'vaizata'];

function getPasswordForUser(username) {
  // Hash the username
  var hashedUsername = sha1(username);
  // Use as an index into a secret list of passwords
  var listIndex = parseInt(hashedUsername.slice(0,4),16) % secretListOfPasswords.length;
  return secretListOfPasswords[listIndex];
}

// Stall a little with added noise
function stall(time, difficulty) {
  msleep(time/difficulty + randn_bm() * difficulty);
}

function verifyPassword(inPassword, correctPassword, difficulty) {
    if (inPassword.length != correctPassword.length) {
        return false;
    }

    stall(BASE_TIME_TO_STALL, difficulty);
 
    for (var i = 0; i < correctPassword.length; i++) {
        stall(BASE_TIME_TO_STALL, difficulty);
        if (inPassword[i] != correctPassword[i]) {
            // prophylactic stalls to make it harder
            for (var j=i;j<correctPassword.length;j++) {
                stall(0, difficulty);
            }
            return false;
        } // if
    } // for
 
    return true;
}


// [START hello_world]
// Say hello!
app.get('/', function(req, res) {
  // Make sure we have a user and a password
  if ((req.query.user != null) && (req.query.password != null)) {
    var secretPassword = "";
    // If there is a nonzero difficulty
    if (req.query.difficulty != null) {
       req.query.difficulty = parseInt(req.query.difficulty);
       // make sure the difficulty is valid
       if (req.query.difficulty < 1) {
           req.query.difficulty = 1;
       }

       secretPassword = getDifficultPasswordForUser(req.query.user,req.query.difficulty);
    } else {
       // low difficulty
       req.query.difficulty = 1;
       secretPassword = getPasswordForUser(req.query.user);
    }

    // Check if it's correct
    var isPasswordCorrect = verifyPassword(req.query.password, secretPassword, req.query.difficulty);

    // Answer the result.
    if (isPasswordCorrect) {
      res.status(200).send('1');
    } else {
      res.status(200).send('0');
    }
  } else { // we didn't get both user and password
    res.status(200).send('Usage: http://127.0.0.1/?user=albert&password=perot[&difficulty=1]');
  }
});
// [END hello_world]

if (module === require.main) {
  // [START server]
  // Start the server
  var server = app.listen(process.env.PORT || 80, function () {
    var host = server.address().address;
    var port = server.address().port;

    console.log('App listening at 127.0.0.1');
  });
  // [END server]
}

module.exports = app;
