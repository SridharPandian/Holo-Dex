using System;
using System.Collections.Generic;

using UnityEngine;
using UnityEngine.UI;

using NetMQ;
using NetMQ.Sockets;

class GestureDetector : MonoBehaviour
{
    // Hand objects
    public OVRHand LeftHand;
    public OVRHand RightHand;
    public OVRSkeleton LeftHandSkeleton;
    public OVRSkeleton RightHandSkeleton;
    public OVRPassthroughLayer PassthroughLayerManager;
    private List<OVRBone> LeftHandFingerBones;
    private List<OVRBone> RightHandFingerBones;

    // Menu and RayCaster GameObjects
    public GameObject MenuButton;
    public GameObject LaserPointer;
    private LineRenderer LineRenderer;

    // Hand Usage indicator
    public RawImage StreamBorder;

    // Stream Enablers
    bool StreamRightHandData = true;
    bool StreamLeftHandData = false;

    // Network enablers
    private NetworkManager netConfig;
    private PushSocket client;
    private string communicationAddress;
    private bool connectionEstablished = false;


    // Starting the server connection
    public void CreateTCPConnection()
     {
        // Check if communication address is available
        communicationAddress = netConfig.getKeypointAddress();
        bool AddressAvailable = !String.Equals(communicationAddress, "tcp://:");

        if (AddressAvailable)
        {
            // Initiate Push Socket
            client = new PushSocket();
            client.Connect(communicationAddress);
            connectionEstablished = true;
        }

        // Setting color to green to indicate control
        StreamBorder.color = Color.red;
    }

    // Start function
    void Start()
     {
        // Getting the Network Config Updater gameobject
        GameObject netConfGameObject = GameObject.Find("NetworkConfigsLoader");
        netConfig = netConfGameObject.GetComponent<NetworkManager>();

        LaserPointer = GameObject.Find("LaserPointer");
        LineRenderer = LaserPointer.GetComponent<LineRenderer>();

        // Initializing the hand skeleton
        LeftHandFingerBones = new List<OVRBone>(LeftHandSkeleton.Bones);
        RightHandFingerBones = new List<OVRBone>(RightHandSkeleton.Bones);

        // Indicating hand stream
        StreamBorder.color = Color.green; // Green for right hand stream
    }


    // Function to serialize the Vector3 List
    public static string SerializeVector3List(List<Vector3> gestureData)
    {
        string vectorString = "";
        foreach (Vector3 vec in gestureData)
            vectorString = vectorString + vec.x + "," + vec.y + "," + vec.z + "|";

        // Clipping last element and using a semi colon instead
        if (vectorString.Length > 0)
            vectorString = vectorString.Substring(0, vectorString.Length - 1) + ":";

        return vectorString;
    }


    // Getting bone information and sending it
    public void SendLeftHandData() 
    {
        // Getting bone positional information
        List<Vector3> leftHandGestureData = new List<Vector3>();
        foreach (var bone in LeftHandFingerBones)
        {
            Vector3 bonePosition = bone.Transform.position;
            leftHandGestureData.Add(bonePosition);
        }

        // Creating a string from the vectors
        string LeftHandDataString = SerializeVector3List(leftHandGestureData);
        LeftHandDataString = "left_hand:" + LeftHandDataString;

        client.SendFrame(LeftHandDataString);
        byte[] recievedToken = client.ReceiveFrameBytes();
    }


    // Getting bone information and sending it
    public void SendRightHandData()
    {
        // Getting bone positional information
        List<Vector3> rightHandGestureData = new List<Vector3>();
        foreach (var bone in RightHandFingerBones)
        {
            Vector3 bonePosition = bone.Transform.position;
            rightHandGestureData.Add(bonePosition);
        }

        // Creating a string from the vectors
        string RightHandDataString = SerializeVector3List(rightHandGestureData);
        RightHandDataString = "right_hand:" + RightHandDataString;

        client.SendFrame(RightHandDataString);
        byte[] recievedToken = client.ReceiveFrameBytes();
    }


    public void StreamPauser()
    {
        // Enabling Passthrough
        //if (LeftHand.GetFingerIsPinching(OVRHand.HandFinger.Pinky))
        //{
        //    PassthroughLayerManager.hidden = !PassthroughLayerManager.hidden;
        //}

        // Switching from Right hand control
        if (LeftHand.GetFingerIsPinching(OVRHand.HandFinger.Middle))
        {
            StreamRightHandData = false;
            StreamLeftHandData = true;
            StreamBorder.color = Color.blue; // Blue for left hand stream
            MenuButton.SetActive(false);
            LineRenderer.enabled = false;
        }

        // Switching from Left hand control
        if (RightHand.GetFingerIsPinching(OVRHand.HandFinger.Middle))
        {
            StreamLeftHandData = false;
            StreamRightHandData = true;
            StreamBorder.color = Color.green; // Green for right hand stream
            MenuButton.SetActive(false);
            LineRenderer.enabled = false;
        }

        // Pausing Stream
        if (LeftHand.GetFingerIsPinching(OVRHand.HandFinger.Ring))
        {
            StreamRightHandData = false;
            StreamLeftHandData = false;
            StreamBorder.color = Color.red; // Red color for no stream
            MenuButton.SetActive(true);
            LineRenderer.enabled = true;
        }
    }

    void Update()
    {
        if (connectionEstablished)
        {
            if (String.Equals(communicationAddress, netConfig.getKeypointAddress()))
            {
                StreamPauser();

                if (StreamLeftHandData)
                    SendLeftHandData();

                if (StreamRightHandData)
                    SendRightHandData();
            }
            else
            {
                connectionEstablished = false;
            }
        } else
        {
            MenuButton.SetActive(true);
            StreamBorder.color = Color.red;
            CreateTCPConnection();
        }
    }
}