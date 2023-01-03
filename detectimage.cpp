#include "detectimage.h"
#include <stdlib.h>


extern QQueue<cv::Mat> videoFrameQueue;
extern QMutex videoMutex;

QQueue<ROI_FRAME> roiFrameQueue;
QMutex detectMutex;

DetectImage::DetectImage()
{

}

bool DetectImage::loadAlgorithmModel(const char *func_name)
{
    cascada = new CascadeClassifier;
    return cascada->load(func_name);
}

void DetectImage::setPause(bool flag)
{
    QMutexLocker locker(&mutex);
    IS_Pause = flag;
}

void DetectImage::setRun(bool flag)
{
    QMutexLocker locker(&mutex);
    AlwaysRun = flag;
}


int DetectImage::detectImageEx3(const char *fun, Mat srcImg, ROI_FRAME &roiFrame)
{
    PyObject *pFun = PyObject_GetAttrString(pDetect, fun); // 获取函数名
    if(!(pFun && PyCallable_Check(pFun)))
    {
        qDebug() << "Failed to get detect function";
        return -1;
    }

    // 将 Mat 类型 转 PyObject* 类型
    PyObject *argList = PyTuple_New(1); /* 创建只有一个元素的元组 */
    npy_intp Dims[3] = {srcImg.rows, srcImg.cols, srcImg.channels()};
    int dim = srcImg.channels() == 1 ? 1 : 3;
    PyObject *PyArray = PyArray_SimpleNewFromData(dim, Dims, NPY_UBYTE, srcImg.data); /* 创建一个数组 */
    PyTuple_SetItem(argList, 0, PyArray); /* 将数组插入元组的第一个位置 */
    // 带传参的函数执行
    PyObject *pRet = PyObject_CallObject(pFun, argList);

    _Py_XDECREF(argList); // 释放内存空间，并检测是否为null，销毁argList时同时也会销毁Pyarray
    _Py_XDECREF(pFun);    // 释放内存空间

    if(!PyTuple_Check(pRet))
    {
        qDebug() << "Failed to get python return value";
        return -2;// 检查返回值是否是元组类型
    }

    PyArrayObject *ret_array = nullptr;
    PyObject *calsses = nullptr;
    PyObject *confs = nullptr;

    int ret = PyArg_UnpackTuple(pRet, "ref", 3, 3, &ret_array ,&calsses, &confs); // 解析返回值
    if(!ret)
    {
        qDebug() << "Failed to unpack tuple";
        _Py_XDECREF(pRet); //  PyObject_CallObject
        return -3;
    }


    int size = PyList_Size(calsses); /* 获取列表长度 */
    for (int i = 0 ; i< size; i++)
    {
        PyObject *val = PyList_GetItem(calsses, i); /* 获取列表中的元素 */
        if(!val) continue;
        char *_class;
        PyArg_Parse(val, "s", &_class);             /* 解析元素 */
        roiFrame.classList.append(QString(_class));
    }

    for (int i = 0 ; i< size; i++)
    {
        PyObject *val = PyList_GetItem(confs, i);
        if(!val) continue;
        char *conf;
        PyArg_Parse(val, "s", &conf);
        roiFrame.confList.append(QString(conf));
    }



    Mat frame = Mat(ret_array->dimensions[0], ret_array->dimensions[1], CV_8UC3, PyArray_DATA(ret_array)).clone(); // 转 Mat 类型
    roiFrame.frame = frame;

    _Py_XDECREF(pRet); //  PyObject_CallObject
    return 0;
}

void DetectImage::run()
{

    loadAlgorithmModel("./haarcascades/haarcascade_frontalface_alt2.xml");   /* 加载算法模型，必须要在同一线程内 */

    vector<Rect> faces;//vector容器存检测到的faces
#if 0
    Mat roi;
    int cnt = 141;
#endif
    while(AlwaysRun)
    {
        if(IS_Pause)
        {
            videoMutex.lock();
            bool isok = videoFrameQueue.isEmpty();
            videoMutex.unlock();

            if(!isok)
            {
                videoMutex.lock();
                Mat srcFrame = videoFrameQueue.dequeue();
                int size = videoFrameQueue.size();
                if (size > 3) videoFrameQueue.clear(); // 针对检测速度较慢的算法，通过对消息队列进行定时清理，以达到实时检测效果
                videoMutex.unlock();

                Mat frame_gray;

                cvtColor(srcFrame, frame_gray, COLOR_BGR2GRAY);//转灰度化，减少运算
                equalizeHist(frame_gray, frame_gray);
                cascada->detectMultiScale(frame_gray, faces, 1.1, 4, CASCADE_DO_ROUGH_SEARCH, Size(70, 70), Size(1000, 1000));

//                qDebug() << "face:" << faces.size();
                ROI_FRAME dstFrame;

                for (int i = 0; i < faces.size(); i++)
                {

                   rectangle(srcFrame, faces[i], Scalar(255, 0, 0), 2, 8, 0);
                }

#if 0 //提取人脸图像进行训练
                if(faces.size() == 1)
                {
                    Mat faceROI = frame_gray(faces[0]);//在灰度图中将圈出的脸所在区域裁剪出
                    resize(faceROI, roi, Size(92, 112));
                    string filename = format("D://AccountFile//Desk//Face//QtPythonDetect//master//bin//datasets//images//%d.jpg", cnt++);
                    imwrite(filename, roi);//存在当前目录下
                }
#endif
                dstFrame.frame = srcFrame;
                faces.clear();
                detectMutex.lock();
                roiFrameQueue.enqueue(dstFrame);
                detectMutex.unlock();
            }
        }
        msleep(100);
    }

}
