import React from 'react';
import {render} from 'react-dom';
import App from './App'
import 'bootstrap/dist/css/bootstrap.css';
import authReducer from "./state/reducersAuth";
import 'babel-polyfill';
import './css/Styles.css'
import {applyMiddleware, combineReducers, compose, createStore} from "redux";
import {createLogger} from "redux-logger";
import thunk from "redux-thunk";
import {crashReporter, logger} from "./state/logger";
import {createBrowserHistory} from "history";
import {connectRouter, routerMiddleware} from "connected-react-router";
import friendRequests from "./state/reducersFriendRequests";


const loggerMiddleware = createLogger(logger, crashReporter);

const history = createBrowserHistory();

const reducer = combineReducers({authentication: authReducer, friendRequests: friendRequests});

const store =  createStore(
    connectRouter(history)(reducer),
    compose(
        applyMiddleware(
        thunk,
        loggerMiddleware,
        routerMiddleware(history)
    ),
    )
);

render(
        <App store={store} history={history} className="App"/>,
    document.getElementById('root')
);
