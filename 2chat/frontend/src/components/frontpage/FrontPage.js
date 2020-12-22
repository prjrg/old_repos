import React, { Component } from 'react';
import '../../css/Styles.css';
import {connect} from "react-redux";
import BodyFrontPage from "./BodyFP";
import FrontPageNav from "./FrontPageNav";


class FrontPage extends Component {
  render() {
    return (
        <div>
          <FrontPageNav/>
        <BodyFrontPage/>
        </div>
    );
  }
}

export default connect()(FrontPage);
