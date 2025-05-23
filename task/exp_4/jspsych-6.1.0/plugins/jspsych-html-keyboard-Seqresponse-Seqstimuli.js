/*
 * Example plugin template
 */

/*
    var NAMEVARIABLE = {
    type: "html-keyboard-Seqresponse",
    stimulus_1 : '<p> TEST </p>',    // The fIRST HTML string to be displayed
    stimulus_2 : '<p> TEST </p>',    // The SECOND HTML string to be displayed
    choices : ['d', 'f', 'j', 'k'],  // The keys the subject is allowed to press to respond to the stimulus.
    stimulus_1_duration : 0.1        // How long to hide the 1st stimulus
    stimulus_2_duration : 0.1        // How long to hide the 2nd stimulus
    trial_duration: 0.5              // How long to show trial before it ends
    max_Resp: 2                      // How long to show trial before it ends
    }
  
*/

jsPsych.plugins["html-keyboard-Seqresponse-Seqstimuli"] = (function() {

  var plugin = {};

  plugin.info = {
    name: "html-keyboard-Seqresponse-Seqstimuli",
    parameters: {
      stimulus_1: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Stimulus',
        default: undefined,
        description: 'The FIRST HTML string to be displayed'
      },
      stimulus_2: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Stimulus',
        default: undefined,
        description: 'The SECOND HTML string to be displayed'
      },
      choices: {
        type: jsPsych.plugins.parameterType.KEYCODE,
        array: true,
        pretty_name: 'Choices',
        default: jsPsych.ALL_KEYS,
        description: 'The keys the subject is allowed to press to respond to the stimulus.'
      },
      stimulus_1_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus duration',
        default: null,
        description: 'How long to hide the stimulus.'
      },
      stimulus_2_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus duration',
        default: null,
        description: 'How long to hide the stimulus.'
      },
      trial_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Trial duration',
        default: null,
        description: 'How long to show trial before it ends.'
      },
      max_Resp: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Trial duration',
        default: null,
        description: 'How long to show trial before it ends.'
      },
      resp_Correct: {
        type: jsPsych.plugins.parameterType.KEYCODE,
        pretty_name: 'Correct response sequence',
        default: null,
        description: 'The response sequence required by the trial'
      }
      }
    }
  

  plugin.trial = function(display_element, trial) {

    var response = {
      rt: null,
      key: null
    };

    var responses_rt = []; 	    // Array for response times  
    var responses_key = []; 	  // Array for response keys 
    var responses_char = ""; 	  // Equivalent char form
    var acc = null;             // performance accuracy

    var html_stim1 = '<div id="jspsych-html-keyboard-response-stimulus">'+trial.stimulus_1+'</div>';
    display_element.innerHTML = html_stim1

    var next_stim = function(){
      var html_stim2 = '<div id="jspsych-html-keyboard-response-stimulus">'+trial.stimulus_2+'</div>';
      display_element.innerHTML = html_stim2
    }; 

    if (trial.stimulus_2 !== null)
      jsPsych.pluginAPI.setTimeout(function() {next_stim();
      }, trial.stimulus_1_duration)


    var end_trial = function() {

      // kill any remaining setTimeout handlers
      jsPsych.pluginAPI.clearAllTimeouts();

      // kill keyboard listeners
      if (typeof keyboardListener !== 'undefined') {
        jsPsych.pluginAPI.cancelKeyboardResponse(keyboardListener);
      }

      // gather the data to store for the trial
      var trial_data = {
        "rt": responses_rt, //response.rt
        "stimulus_1": trial.stimulus_1,
        "stimulus_2": trial.stimulus_2,
        "key_press": responses_key,//response.key
        "key_press_char": responses_char,
        "accuracy": acc,
      };

      // clear the display
      display_element.innerHTML = '';

      // move on to the next trial
      jsPsych.finishTrial(trial_data);
    };

    // function to handle responses by the subject
    var after_response = function(info) {

      // after a valid response, the stimulus will have the CSS class 'responded'
      // which can be used to provide visual feedback that a response was recorded
      display_element.querySelector('#jspsych-html-keyboard-response-stimulus').className += ' responded';

      if (responses_key.length <= trial.max_Resp) {
        response = info;
        responses_rt.push(response.rt)
        responses_key.push(response.key)
        responses_char = responses_char+jsPsych.pluginAPI.convertKeyCodeToKeyCharacter(response.key)
        if(responses_key.length == trial.max_Resp){
          if (trial.resp_Correct != null){
            responses_char = responses_char.toLowerCase().split("");
            responses_char = responses_char.sort();
            responses_char = responses_char.join("");
            resp_Correct = trial.resp_Correct.sort();
            resp_Correct = resp_Correct.join("");
            resp_Correct = resp_Correct.toLowerCase();
            acc = responses_char == resp_Correct;
          }}
          }
        }
    

    // start the response listener
    if (trial.choices != jsPsych.NO_KEYS) {
        var keyboardListener = jsPsych.pluginAPI.getKeyboardResponse({
            callback_function: after_response,
            valid_responses: trial.choices,
            rt_method: 'performance',
            persist: true,
            allow_held_key: false
        });
        }
    
    // hide stimulus if stimulus_duration is set
    //if (trial.stimulus_duration !== null) {
    //jsPsych.pluginAPI.setTimeout(function() {
    //    display_element.querySelector('#jspsych-html-keyboard-response-stimulus').style.visibility = 'hidden';
    //}, trial.stimulus_duration);
    //}
  

    // end trial if trial_duration is set
    if (trial.trial_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        end_trial();
      }, trial.trial_duration);
    }

  }
  return plugin;
})();
