import React, {Component} from 'react';
import {Jumbotron} from "reactstrap";


class Title extends Component {
    constructor(props){
        super(props);
        this.state = {
            email: "diana8vicente@gmail.com"
        }
    }

    render() {
        return(<div>
                <Jumbotron>
                    <h1>Logo Designs</h1>
                    <h2>Made by Diana Vicente</h2>
                    <h4>Contact her through <a href={"mailto:" + this.state.email}/>email</h4>
                </Jumbotron>
            </div>
        );
    }
}

export default Title;