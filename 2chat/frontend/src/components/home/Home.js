import React, { Component } from 'react';
import '../../css/Styles.css';
import {connect} from "react-redux";
import HomeNav from "./HomeNav";


class Home extends Component {
    render() {
        return (
            <div>
                <HomeNav/>

            </div>
        );
    }
}

export default connect()(Home);
