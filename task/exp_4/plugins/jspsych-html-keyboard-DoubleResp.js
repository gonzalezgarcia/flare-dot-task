/*
 * Example plugin template
 */

/*
    var NAMEVARIABLE = {
    type: "html-keyboard-Seqresponse",
    stimulus : '<p> TEST </p>',      // The HTML string to be displayed
    choices : ['d', 'f', 'j', 'k'],  // The keys the subject is allowed to press to respond to the stimulus.
    stimulus_duration : 0.1          // How long to hide the stimulus
    trial_duration: 0.5              // How long to show trial before it ends
    max_Resp: 2                      // How long to show trial before it ends
    }
  
*/

jsPsych.plugins["html-keyboard-DoubleResp"] = (function() {

  var plugin = {};

  plugin.info = {
    name: "html-keyboard-DoubleResp     ",
    parameters: {
      stimulus: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Stimulus',
        default: undefined,
        description: 'The HTML string to be displayed'
      },
      choices: {
        type: jsPsych.plugins.parameterType.KEYCODE,
        array: true,
        pretty_name: 'Choices',
        default: jsPsych.ALL_KEYS,
        description: 'The keys the subject is allowed to press to respond to the stimulus.'
      },
      stimulus_duration: {
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
      response_delay:{
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Delay between responses',
          default: null,
          description: 'How many ms should elapse between responses to be considered part of the same chunk'
      }
      }
    }
  

    plugin.trial = function(display_element, trial) {

        var new_html = '<div id="jspsych-html-keyboard-response-stimulus">'+trial.stimulus+'</div>';

        // draw
        display_element.innerHTML = new_html;

        // store response
        var response = {
        rt: null,
        key: null
        };
        
        // Array for response times 
        var responses_rt = []; 		// Timing from the stimulus presentation
        
        // Array for response keys 
        var responses_key = []; 		// JS numeric code
        var responses_char = ""; 	// Equivalent char form
        // function to end trial when it is time
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
            "stimulus": trial.stimulus,
            "key_press": responses_key,//response.key
            "key_press_char": responses_char,
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
                responses_char = responses_char+jsPsych.pluginAPI.convertKeyCodeToKeyCharacter(response.key)}        
                // end trial if trial_duration is set
                if (trial.response_delay !== null) {
                    jsPsych.pluginAPI.setTimeout(function() {
                    end_trial();
                    }, trial.response_delay);
                }
            if(responses_key.length == trial.max_Resp){   
                jsPsych.pluginAPI.cancelKeyboardResponse(keyboardListener)
                responses_char = responses_char.split("");
                responses_char = responses_char.sort();
                responses_char = responses_char.join("");         
                if (trial.response_ends_trial) {
                end_trial();
            }
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
        if (trial.stimulus_duration !== null) {
        jsPsych.pluginAPI.setTimeout(function() {
            display_element.querySelector('#jspsych-html-keyboard-response-stimulus').style.visibility = 'hidden';
        }, trial.stimulus_duration);
        }
      

        // end trial if trial_duration is set
        if (trial.trial_duration !== null) {
          jsPsych.pluginAPI.setTimeout(function() {
            end_trial();
          }, trial.trial_duration);
        }

      }
  return plugin;
})();
