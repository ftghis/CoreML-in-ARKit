//
//  File.swift
//  
//
//  Created by Ghislain Fouodji Tasse on 9/27/22.
//

import Foundation

import Vision
import CoreImage

@available(iOS 11.0, *)
extension Doto {

    @available(iOS 11.0, *)
    public class DTVision {

        var modelName: String
        var config: (version: String, retrain_on_data_interval: Int)
        var requests = [VNRequest]()

        public init(modelName: String, config: (version: String, retrain_on_data_interval: Int) = (version: "latest", retrain_on_data_interval: 1000)) {
            self.modelName = modelName
            self.config = config
        }

       public func setUpPrediction(completionHandler: VNRequestCompletionHandler?) {
           let url = URL(string: "https://github.com/hanleyweng/CoreML-in-ARKit/raw/73465d1e6c9f4110d8213f39ac31bd1970fef4d0/CoreML%20in%20ARKit/" + self.modelName + ".mlmodel")!
           setUpModelFromURL(url: url, completionHandler: completionHandler)
       }

        public func predict(id: String, pixbuff : CVPixelBuffer?) {
            if pixbuff == nil { return }

            let ciImage = CIImage(cvPixelBuffer: pixbuff!)
            let imageRequestHandler = VNImageRequestHandler(ciImage: ciImage, options: [:])

            // Run Image Request
            do {
                try imageRequestHandler.perform(requests)
            } catch {
                print(error)
            }

        }

        public func data(correction: (id: String, classification: String)) {
            //send in batches of 100
        }

        private func setUpModelFromURL(url: URL, completionHandler: VNRequestCompletionHandler?) {

            let downloadTask = URLSession.shared.downloadTask(with: url) { urlOrNil, responseOrNil, errorOrNil in
                // check for and handle errors:
                // * errorOrNil should be nil
                // * responseOrNil should be an HTTPURLResponse with statusCode in 200..<299

                guard let fileURL = urlOrNil else { return }

                do {
                    let compiledModelURL = try MLModel.compileModel(at: fileURL)
                    let model = try MLModel(contentsOf: compiledModelURL)
                    self.setUpCoreML(model: model, completionHandler: completionHandler)

//                    let documentsURL = try
//                        FileManager.default.url(for: .documentDirectory,
//                                                in: .userDomainMask,
//                                                appropriateFor: nil,
//                                                create: false)
//                    let savedURL = documentsURL.appendingPathComponent(fileURL.lastPathComponent)
//                    try FileManager.default.moveItem(at: fileURL, to: savedURL)
                } catch {
                    print ("file error: \(error)")
                }
            }
            downloadTask.resume()
        }

        private func setUpCoreML(model: MLModel, completionHandler: VNRequestCompletionHandler?) {
            // Set up Vision Model
           guard let selectedModel = try? VNCoreMLModel(for: model) else { // (Optional) This can be replaced with other models on https://developer.apple.com/machine-learning/
                fatalError("Could not load model. Ensure model has been drag and dropped (copied) to XCode Project from https://developer.apple.com/machine-learning/ . Also ensure the model is part of a target (see: https://stackoverflow.com/questions/45884085/model-is-not-part-of-any-target-add-the-model-to-a-target-to-enable-generation ")
            }

            // Set up Vision-CoreML Request
           let request = VNCoreMLRequest(model: selectedModel, completionHandler: completionHandler)
            request.imageCropAndScaleOption = VNImageCropAndScaleOption.centerCrop // Crop from centre of images and scale to appropriate size.
           requests = [request]
        }
    }
}
