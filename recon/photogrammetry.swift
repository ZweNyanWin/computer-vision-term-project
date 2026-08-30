// Multi-view reconstruction with Apple's Object Capture.
//
// Swift rather than Python for one reason: the engine lives in RealityKit and
// Apple exposes it to code only -- there is no shipped command and no Python
// binding. Compile once, then use it like any other tool.
//
//   swiftc -O recon/photogrammetry.swift -o recon/photogrammetry
//   ./recon/photogrammetry recon/input_ring_low recon/frog.usdz medium
//
// sampleOrdering is .sequential because the photographs are one ordered
// turntable ring. That is the same restriction the rest of the project already
// applies to matching, and for the same reason: telling the engine the frames
// are consecutive lets it match neighbours instead of searching every pair.
//
// Expect the top of the frog to reconstruct badly. Every photograph in this
// ring was taken at eye level, so nothing ever looked down on the carved back.
// That is a coverage limit of the capture, not of the method, and is worth
// reporting rather than hiding.

import Foundation
import RealityKit

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(("ERROR: " + message + "\n").data(using: .utf8)!)
    exit(1)
}

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    fail("usage: photogrammetry <input-folder> <output.usdz> [preview|reduced|medium|full|raw]")
}

let inputURL = URL(fileURLWithPath: arguments[1])
let outputURL = URL(fileURLWithPath: arguments[2])
let detailName = arguments.count > 3 ? arguments[3] : "medium"

let detailLevels: [String: PhotogrammetrySession.Request.Detail] = [
    "preview": .preview, "reduced": .reduced, "medium": .medium,
    "full": .full, "raw": .raw,
]
guard let detail = detailLevels[detailName] else {
    fail("unknown detail '\(detailName)'; expected preview, reduced, medium, full or raw")
}

guard PhotogrammetrySession.isSupported else {
    fail("Object Capture is not supported on this machine")
}

let frameCount = (try? FileManager.default.contentsOfDirectory(atPath: inputURL.path))?
    .filter { $0.lowercased().hasSuffix(".jpeg") || $0.lowercased().hasSuffix(".jpg") }.count ?? 0
print("input:  \(inputURL.path)  (\(frameCount) photographs)")
print("output: \(outputURL.path)")
print("detail: \(detailName), sampleOrdering: sequential\n")

var configuration = PhotogrammetrySession.Configuration()
configuration.sampleOrdering = .sequential
configuration.featureSensitivity = .high   // matte carved wood, fine surface relief

let started = Date()
let finished = DispatchSemaphore(value: 0)
var failureMessage: String?
var stitchingIncomplete = false

do {
    let session = try PhotogrammetrySession(input: inputURL, configuration: configuration)

    Task {
        do {
            for try await output in session.outputs {
                switch output {
                case .requestProgressInfo(_, _):
                    break
                case .requestProgress(_, let fraction):
                    let percent = Int(fraction * 100)
                    if percent % 5 == 0 {
                        print("  \(percent)%  (\(Int(Date().timeIntervalSince(started)))s)")
                    }
                case .requestComplete(_, _):
                    print("\nwrote \(outputURL.path)")
                case .requestError(_, let error):
                    failureMessage = "request failed: \(error.localizedDescription)"
                case .inputComplete:
                    print("  all photographs read")
                case .invalidSample(let id, let reason):
                    print("  ! photograph \(id) rejected: \(reason)")
                case .skippedSample(let id):
                    print("  ! photograph \(id) skipped")
                case .automaticDownsampling:
                    print("  ! the engine downsampled the input")
                case .stitchingIncomplete:
                    // Surface the partial mesh rather than swallowing it. With a
                    // single eye-level ring this is the expected outcome at the
                    // top of the object, and it belongs in the report.
                    stitchingIncomplete = true
                    print("  ! STITCHING INCOMPLETE - the mesh is partial")
                case .processingComplete:
                    finished.signal()
                case .processingCancelled:
                    failureMessage = "processing was cancelled"
                    finished.signal()
                @unknown default:
                    break
                }
            }
        } catch {
            failureMessage = "session output stream failed: \(error.localizedDescription)"
            finished.signal()
        }
    }

    try session.process(requests: [.modelFile(url: outputURL, detail: detail)])
    finished.wait()
} catch {
    fail("could not start the session: \(error.localizedDescription)")
}

if let failureMessage {
    fail(failureMessage)
}
if stitchingIncomplete {
    print("\nNOTE: the engine reported stitchingIncomplete, so the mesh does not")
    print("      close. With one eye-level ring the gap is the top of the object.")
}
print("elapsed: \(Int(Date().timeIntervalSince(started)))s")
