import {Form, FormGroup, Label, Input, Button} from 'reactstrap';
import React, {Component} from 'react';
import PropTypes from 'prop-types';
import {login} from "../../state/ActionsAuth";
import {connect} from "react-redux";
import {withRouter} from 'react-router-dom';
import {push} from "react-router-redux";

class LoginForm extends Component {
    constructor(props){
        super(props);
        this.state = {
            username: this.props.username,
            password: ""
        };
        this.handleSubmit = this.handleSubmit.bind(this);
        this.handleChange = this.handleChange.bind(this);

    }

    componentWillReceiveProps(nextProps){
        if(nextProps.authenticated){
            this.props.loggedIn();
        }
    }

    handleSubmit(e){
        e.preventDefault();

        const {username, password} = this.state;
        if(username && password){
            this.props.submit(username, password);
        }

    }

    handleChange(e){
        const {name, value} = e.target;
        this.setState(prevState => ({...prevState, [name]: value}));
    }

    render(){
        let {username, password} = this.state;
        return (
                        <Form inline onSubmit={this.handleSubmit}>
                            <FormGroup>
                                <Label for="uName" hidden>Username</Label>
                                <Input className="input-chat-login" name="username" placeholder={username === "" ? "Username or Email" : username} value={username} onChange={this.handleChange}/>
                            </FormGroup>
                            {' '}
                            <FormGroup>
                                <Label for="uPassword" hidden>Password</Label>
                                <Input className="input-chat-login" type="password" name="password" placeholder="Enter Password" value={password} onChange={this.handleChange}/>
                            </FormGroup>
                            {' '}
                            <Button className="submit-chat">Login</Button>
                        </Form>
        )
    }
}

LoginForm.propTypes = {
    loggingIn: PropTypes.bool.isRequired,
    username: PropTypes.string.isRequired,
    loginError: PropTypes.number.isRequired,
    authenticated: PropTypes.bool.isRequired,
    submit: PropTypes.func.isRequired,
    loggedIn: PropTypes.func.isRequired
};

function mapStateToProps(state){
    const {authentication} = state;
    const {username, login, authenticated} = authentication;
    const {loggingIn, loginError} = login;
    return {
        loggingIn,
        username,
        loginError,
        authenticated
    };
}

const mapDispatchToProps = dispatch => {
    return {
        submit: (username, password) => {dispatch(login(username, password))},
        loggedIn: () => {dispatch(push("/"))},
    }
};

const Login = connect(mapStateToProps, mapDispatchToProps)(LoginForm);

export default withRouter(Login);