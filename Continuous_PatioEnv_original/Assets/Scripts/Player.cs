using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

public class Player : Agent
{
    // 各種パラメータ
    public Vector3 initPos;
    public float maxMotorTorque;

    // センサ
    public GameObject groundSensor;
    
    // LED
    public GameObject ledFl;
    public GameObject ledFr;
    public GameObject ledRear;
    
    // ホイール
    public WheelCollider leftWheel;
    public WheelCollider rightWheel;
    
    private Rigidbody _playerRb;
    private float _fitness;
    private int _cptCount;
    private int _prevCptCount;
    private float _collisionValue;
    
    public override void Initialize()
    {
        var envParams = Academy.Instance.EnvironmentParameters;
        
        _playerRb = GetComponent<Rigidbody>();
        MaxStep = (int) envParams.GetWithDefault("time_steps", 1000.0f) * (int) Time.timeScale;
    }

    public override void OnEpisodeBegin()
    {
        _playerRb.velocity = Vector3.zero;
        
        // 初期化処理
        transform.position = initPos;
        transform.rotation = Quaternion.Euler(0.0f, Random.Range(0.0f, 360.0f), 0.0f);;

        _fitness = 0;
        _cptCount = 0;
        _prevCptCount = 0;
        groundSensor.GetComponent<GroundSensor>().cptCount = 0;
    }

    private void OnCollisionStay(Collision other)
    {
        if (other.gameObject.CompareTag("Robot") || other.gameObject.CompareTag("Wall"))
        {
            _collisionValue = 1.0f;
        }
    }

    private void OnCollisionExit(Collision other)
    {
        if (other.gameObject.CompareTag("Robot") || other.gameObject.CompareTag("Wall"))
        {
            _collisionValue = 0.0f;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(_fitness);
        sensor.AddObservation(_playerRb.position.x);
        sensor.AddObservation(_playerRb.position.z);
        sensor.AddObservation(_collisionValue);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        _prevCptCount = _cptCount;
        
        // アクションを取得
        float leftWheelInput = maxMotorTorque * actions.ContinuousActions[0];
        float rightWheelInput = maxMotorTorque * actions.ContinuousActions[1];
        float ledFrontInput = actions.ContinuousActions[2];
        float ledRearInput = actions.ContinuousActions[3];

        // 左車輪の回転
        float leftTorqueNow = leftWheel.motorTorque;
        if (leftTorqueNow * leftWheelInput < 0 || leftWheelInput == 0) {
            leftWheel.motorTorque = 0;
            leftWheel.brakeTorque = Mathf.Abs(leftTorqueNow - leftWheelInput);
        } else {
            if (Mathf.Abs(leftWheel.rpm) < 109.0f) {
                leftWheel.motorTorque = Mathf.Lerp(leftTorqueNow, leftWheelInput, Time.deltaTime);
            } else {
                leftWheel.motorTorque = 0;
            }
            leftWheel.brakeTorque = 0;
        }
        
        // 右車輪の回転
        float rightTorqueNow = rightWheel.motorTorque;
        if (rightTorqueNow * rightWheelInput < 0 || rightWheelInput == 0) {
            rightWheel.motorTorque = 0;
            rightWheel.brakeTorque = Mathf.Abs(rightTorqueNow - rightWheelInput);
        } else {
            if (Mathf.Abs(rightWheel.rpm) < 109.0f) {
                rightWheel.motorTorque = Mathf.Lerp (rightTorqueNow, rightWheelInput, Time.deltaTime);
            } else {
                rightWheel.motorTorque = 0;
            }
            rightWheel.brakeTorque = 0;
        }
        
        // 前方LEDの点灯・消灯
        if (ledFrontInput > 0.0f)
        {
            ledFl.GetComponent<Renderer>().material.color = Color.blue;
            ledFr.GetComponent<Renderer>().material.color = Color.blue;
            ledFl.GetComponent<Renderer> ().material.EnableKeyword("_EMISSION");
            ledFl.GetComponent<Renderer>().material.SetColor("_EmissionColor", new Color(0.0f, 0.0f, 5.0f) * 1);
            ledFr.GetComponent<Renderer> ().material.EnableKeyword("_EMISSION");
            ledFr.GetComponent<Renderer>().material.SetColor("_EmissionColor", new Color(0.0f, 0.0f, 5.0f) * 1);
        }
        else
        {
            ledFl.GetComponent<Renderer>().material.color =  new Color (0.5f, 0.5f, 0.5f, 0);
            ledFr.GetComponent<Renderer>().material.color =  new Color (0.5f, 0.5f, 0.5f, 0);
            ledFl.GetComponent<Renderer> ().material.EnableKeyword("_EMISSION");
            ledFl.GetComponent<Renderer> ().material.SetColor("_EmissionColor",new Color(0.5f, 0.5f, 0.5f)*1);
            ledFr.GetComponent<Renderer> ().material.EnableKeyword("_EMISSION");
            ledFr.GetComponent<Renderer> ().material.SetColor("_EmissionColor",new Color(0.5f, 0.5f, 0.5f)*1);
        }

        // 後方LEDの点灯・消灯
        if (ledRearInput > 0.0f)
        {
            ledRear.GetComponent<Renderer>().material.color = Color.red;
            ledRear.GetComponent<Renderer>().material.EnableKeyword("_EMISSION");
            ledRear.GetComponent<Renderer>().material.SetColor("_EmissionColor", new Color(5.0f, 0.0f, 0.0f) * 1);
        }
        else
        {
            ledRear.GetComponent<Renderer>().material.color =  new Color (0.5f, 0.5f, 0.5f, 0);
            ledRear.GetComponent<Renderer>().material.EnableKeyword("_EMISSION");
            ledRear.GetComponent<Renderer>().material.SetColor("_EmissionColor", new Color(0.5f, 0.5f, 0.5f) * 1);
        }

        _cptCount = groundSensor.GetComponent<GroundSensor>().cptCount;
        if (_cptCount > _prevCptCount)
        {
            AddReward(1.0f);
            _fitness++;
        }
    }

    // 手動テスト用
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // デフォルトの入力
        float leftWheelTorque = Input.GetKey(KeyCode.LeftArrow) ? 0.1f : 0.0f;
        float rightWheelTorque = Input.GetKey(KeyCode.RightArrow) ? 0.1f : 0.0f;
        float frontLed =Input.GetKey(KeyCode.UpArrow) ? 1.0f : 0.0f;
        float rearLed = Input.GetKey(KeyCode.DownArrow) ? 1.0f : 0.0f;
        
        // 入力をエージェントのアクションに割り当て
        ActionSegment<float> continuousAct = actionsOut.ContinuousActions;
        continuousAct[0] = leftWheelTorque;
        continuousAct[1] = rightWheelTorque;
        continuousAct[2] = frontLed;
        continuousAct[3] = rearLed;
    }
}