import React, {Component} from 'react'
import {
    Jumbotron
} from 'reactstrap'
import '../../css/Styles.css'


class BodyFrontPage extends Component {

    render(){

        return (
            <div>
                <Jumbotron className="Home-body">
                    <h1 className="home">You're Welcome To Chat!</h1>
                    <hr className="chat-line" />
                    <p className="home">On the Platform, Message Anytime! Anyone!</p>
                </Jumbotron>
            </div>

        )
    }
}



export default BodyFrontPage;