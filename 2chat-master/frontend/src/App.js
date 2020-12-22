import React from 'react';
import {Route, Switch} from 'react-router-dom';
import {Provider} from "react-redux";
import SignUp from "./components/auth/SignUp";
import Home from "./components/home/Home";
import FrontPage from "./components/frontpage/FrontPage";
import {ConnectedRouter} from 'react-router-redux';
import Authorization from "./components/Authorization";

let condRender = (auth, notAuth) => (() => (<Authorization CompAuth={auth} CompOpt={notAuth}/>));

const App = ({store, history}) => (


    <Provider store={store}>
        <ConnectedRouter history={history}>
            <Switch>
                <Route exact path="/" render={condRender(Home, FrontPage)}/>
                <Route exact path="/signup" component={SignUp}/>
            </Switch>
        </ConnectedRouter>
    </Provider>
        );

export default App;