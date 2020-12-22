import {
    Button, Form, FormGroup, Label, Input, Col, Container, Row, Navbar, NavbarBrand
} from 'reactstrap';
import React, {Component} from 'react';
import PropTypes from 'prop-types';
import {connect} from "react-redux";
import {defaultRegister, signUp} from "../../state/ActionsAuth";
import {withRouter} from 'react-router-dom';

import '../../css/Styles.css';
import {Link} from "react-router-dom";
import {push} from "react-router-redux";

class SignUpForm extends Component {
    constructor(props){
        super(props);
        this.state = {
            username: "",
            email: "",
            password: "",
            password2: "",
            registered: false,
            isOpen: false
        };
        this.handleSubmit = this.handleSubmit.bind(this);
        this.handleChange = this.handleChange.bind(this);
        this.toggle = this.toggle.bind(this);
    }

    toggle() {
        this.setState({
            isOpen: !this.state.isOpen
        });
    }

    componentWillReceiveProps(nextProps){
        if(nextProps.registered){
            this.props.success();
        }
    }

    componentWillUnmount(){
        this.props.clearRegister();
    }

    handleSubmit(e){
        e.preventDefault();

        const {username, email, password, password2} = this.state;
        if(username && email && password && password2){
            this.props.submit(username, email, password, password2);
        }

    }

    handleChange(e){
        const {name, value} = e.target;
        this.setState(prevState => ({...prevState, [name]: value}));
    }

    render(){

        const {username, email, password, password2} = this.state;
        return (
                <div>
                    <Navbar className="navbar-dark bg-dark" expand="md">
                        <NavbarBrand tag={Link} to="/" id="logonav">2Chat</NavbarBrand>
                    </Navbar>
                    <div className="centerOnPage">
                        <Container>
                            <Row>
                                <Col xs={12} sm={12} md={8} lg={6}>
                                <p className="header">Please, set up Your Account</p>
                                </Col>
                            </Row>
                            <Row>
                                <Col xs={12} sm={12} md={8} lg={6}>
                                        <Form>
                                            <FormGroup row>
                                                <Label xs={3} sm={3} md={3} for="username" className="label-chat">Username</Label>
                                                <Col xs={9} sm={9} md={9}>
                                                    <Input className="input-chat" name="username" id="username" placeholder="Choose your username" value={username} onChange={this.handleChange}/>
                                                </Col>
                                            </FormGroup>
                                            <FormGroup row>
                                                <Label xs={3} sm={3} md={3} className="label-chat" for="uEmail">Email</Label>
                                                <Col xs={9} sm={9} md={9}>
                                                    <Input className="input-chat" type="email" name="email" id="uEmail" placeholder="Enter Your Email Address" value={email} onChange={this.handleChange}/>
                                                </Col>
                                            </FormGroup>
                                            <FormGroup row>
                                                <Label xs={3} sm={3} md={3} className="label-chat" for="uPassword">Password</Label>
                                                <Col xs={9} sm={9} md={9}>
                                                    <Input type="password" className="input-chat" name="password" id="uPassword" placeholder="Enter Password" value={password} onChange={this.handleChange}/>
                                                </Col>
                                            </FormGroup>
                                            <FormGroup row>
                                                <Label xs={3} sm={3} md={3} className="label-chat" for="uPassword2"> </Label>
                                                <Col xs={9} sm={9} md={9}>
                                                    <Input type="password" className="input-chat" name="password2" id="uPassword" placeholder="Re-enter Password" value={password2} onChange={this.handleChange}/>
                                                </Col>
                                            </FormGroup>
                                            <Row>
                                            <Col xs={3} sm={3} md={3}>
                                            </Col>
                                            <Col xs={9} sm={9} md={9}>
                                                <Button block className="form-btn-chat" onClick={this.handleSubmit}>Sign up</Button>
                                            </Col>
                                            </Row>
                                        </Form>
                                </Col>

                            </Row>
                        </Container>
                    </div>
                </div>
        )
    }
}

SignUpForm.propTypes = {
    submit: PropTypes.func.isRequired,
    clearRegister: PropTypes.func.isRequired,
    registerError: PropTypes.number.isRequired,
    registering: PropTypes.bool.isRequired,
    registered: PropTypes.bool.isRequired,
    success: PropTypes.func.isRequired
};

const mapStateToProps = (state) => {
    const {authentication} = state;
    const {register} = authentication;
    const {registering, registerError, registered} = register;
    return {
        registering,
        registerError,
        registered
    };
};

const mapDispatchToProps = dispatch => {
    return {
        submit: (...args) => {dispatch(signUp(...args))},
        success: () => {dispatch(push("/"))},
        clearRegister: () => {dispatch(defaultRegister())}
    }
};

const SignUp = connect(mapStateToProps,mapDispatchToProps)(SignUpForm);

export default withRouter(SignUp);